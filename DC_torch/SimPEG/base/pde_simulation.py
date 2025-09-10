import torch
import numpy as np
import scipy.sparse as sp
from discretize.utils import Zero
from ..simulation import BaseSimulation
from .. import props
from scipy.constants import mu_0
from SimPEG.config import SimpegConfig

def __inner_mat_mul_op(M, u, v=None, adjoint=False):
    u = np.squeeze(u)
    if v is not None:
        if v.ndim > 1:
            v = np.squeeze(v)
        if u.ndim > 1:
            # u has multiple fields
            if v.ndim == 1:
                v = v[:, None]
        else:
            if v.ndim > 1:
                u = u[:, None]
        if v.ndim > 2:
            u = u[:, None, :]
        if adjoint:
            if u.ndim > 1 and u.shape[-1] > 1:
                return M.T * (u * v).sum(axis=-1)
            return M.T * (u * v)
        if u.ndim > 1 and u.shape[1] > 1:
            return np.squeeze(u[:, None, :] * (M * v)[:, :, None])
        return u * (M * v)
    else:
        if u.ndim > 1:
            UM = sp.vstack([sp.diags(u[:, i]) @ M for i in range(u.shape[1])])
        else:
            U = sp.diags(u, format="csr")
            UM = U @ M
        if adjoint:
            return UM.T
        return UM



def with_property_mass_matrices(property_name):
    """
    This decorator will automatically populate all of the property mass matrices.

    Given the property "prop", this will add properties and functions to the class
    representing all of the possible mass matrix operations on the mesh.

    For a given property, "prop", they will be named:

    * MccProp
    * MccPropDeriv
    * MccPropI
    * MccPropIDeriv

    and so on for each "Mcc", "Mn", "Mf", and "Me".
    """

    def decorator(cls):
        arg = property_name.lower()
        arg = arg[0].upper() + arg[1:]
        
        @staticmethod
        def torch_diags(diagonals, offsets=0, shape=None, dtype=None):
            device = SimpegConfig().device
            # Procesar diagonales y offsets
            if not isinstance(diagonals, (list, tuple)):
                diagonals = [diagonals]
            if not isinstance(offsets, (list, tuple)):
                offsets = [offsets] * len(diagonals)
            elif len(offsets) != len(diagonals):
                raise ValueError("El número de diagonales y offsets debe coincidir.")
            
            # Determinar la forma de la matriz
            if shape is None:
                n = 0
                for d, k in zip(diagonals, offsets):
                    d_tensor = torch.as_tensor(d)
                    len_d = 1 if d_tensor.ndim == 0 else len(d_tensor)
                    required_n = len_d + abs(k)
                    n = max(n, required_n)
                shape = (n, n)
            else:
                if len(shape) != 2:
                    raise ValueError("La forma debe ser una tupla de dos enteros (filas, columnas).")
                m, n = shape
            
            m, n = shape
            all_indices = []
            all_values = []
            
            for d, k in zip(diagonals, offsets):
                # Calcular la longitud requerida para la diagonal
                if k >= 0:
                    L = min(m, n - k)
                else:
                    L = min(n, m + k)
                
                # Procesar la diagonal (escalar o array)
                d_tensor = d
                if d_tensor.ndim == 0:  # Escalar
                    values = torch.full((L,), d_tensor.item(), dtype=dtype)
                else:
                    d_flat = d_tensor.flatten()
                    if len(d_flat) > L:
                        values = d_flat[:L]
                    else:
                        pad_size = L - len(d_flat)
                        
                        values = torch.cat([d_flat, torch.zeros(pad_size, dtype=dtype, device=device)])
                
                # Generar índices de filas y columnas
                if k >= 0:
                    rows = torch.arange(L, dtype=torch.long)
                    cols = rows + k
                else:
                    cols = torch.arange(L, dtype=torch.long)
                    rows = cols + (-k)
                
                # Asegurar que los índices estén dentro de los límites
                rows = torch.clamp(rows, 0, m - 1)
                cols = torch.clamp(cols, 0, n - 1)

                # if SimpegConfig().all_dense:
                #     result[(rows, cols)] += values
                #     return result

                all_indices.append(torch.stack([rows, cols]))
                all_values.append(values)
            
            # Concatenar todos los índices y valores
            if not all_indices:
                return torch.sparse_coo_tensor(size=shape, dtype=dtype, device=device)
            
            indices = torch.cat(all_indices, dim=1).to(device)
            values = torch.cat(all_values)
            
            # Crear el tensor disperso
            return torch.sparse_coo_tensor(indices=indices, values=values, size=shape, dtype=dtype)

        setattr(cls, "torch_diags", torch_diags)
        @property
        def Mcc_prop(self):
            """
            Cell center property inner product matrix.
            """
            stash_name = f"_Mcc_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = sp.diags(self.mesh.cell_volumes * prop, format="csr")
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mcc{arg}", Mcc_prop)

        @property
        def Mn_prop(self):
            """
            Node property inner product matrix.
            """
            stash_name = f"_Mn_{arg}"
            
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())

                if isinstance(prop, torch.Tensor):
                    prop = prop.to(SimpegConfig().dtype).to(SimpegConfig().device)
                    vol = torch.tensor(self.mesh.cell_volumes, dtype=SimpegConfig().dtype, device=SimpegConfig().device)

                    aveN2CCT = self.mesh.aveN2CC.T
                    if not isinstance(aveN2CCT, torch.Tensor):
                        coo_mat = aveN2CCT.tocoo()
                        row = torch.tensor(coo_mat.row, dtype=torch.long, device=SimpegConfig().device)
                        col = torch.tensor(coo_mat.col, dtype=torch.long, device=SimpegConfig().device)
                        values = torch.tensor(coo_mat.data, dtype=SimpegConfig().dtype, device=SimpegConfig().device) 

                        indices = torch.stack([row, col], dim=0)

                        forma = coo_mat.shape
                        aveN2CCT = torch.sparse_coo_tensor(indices, values, forma, dtype=SimpegConfig().dtype, device=SimpegConfig().device)

                    M_prop = self.torch_diags(torch.sparse.mm(aveN2CCT,(vol * prop).unsqueeze(1)))

                    setattr(self, stash_name, M_prop)

                else:
                    vol = self.mesh.cell_volumes
                    M_prop = sp.diags(self.mesh.aveN2CC.T * (vol * prop), format="csr")
                    setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mn{arg}", Mn_prop)

        @property
        def Mf_prop(self):
            """
            Face property inner product matrix.
            """
            stash_name = f"_Mf_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                
                M_prop = self.mesh.get_face_inner_product(model=prop)
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mf{arg}", Mf_prop)

        @property
        def Me_prop(self):
            """
            Edge property inner product matrix.
            """
            stash_name = f"_Me_{arg}"
            
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())

                M_prop = self.mesh.get_edge_inner_product(model=prop)
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Me{arg}", Me_prop)

        @property
        def MccI_prop(self):
            """
            Cell center property inner product inverse matrix.
            """
            stash_name = f"_MccI_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = sp.diags(1.0 / (self.mesh.cell_volumes * prop), format="csr")
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mcc{arg}I", MccI_prop)

        @property
        def MnI_prop(self):
            """
            Node property inner product inverse matrix.
            """
            stash_name = f"_MnI_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                vol = self.mesh.cell_volumes
                M_prop = sp.diags(
                    1.0 / (self.mesh.aveN2CC.T * (vol * prop)), format="csr"
                )
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mn{arg}I", MnI_prop)

        @property
        def MfI_prop(self):
            """
            Face property inner product inverse matrix.
            """
            stash_name = f"_MfI_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = self.mesh.get_face_inner_product(
                    model=prop, invert_matrix=True
                )
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mf{arg}I", MfI_prop)

        @property
        def MeI_prop(self):
            """
            Edge property inner product inverse matrix.
            """
            stash_name = f"_MeI_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = self.mesh.get_edge_inner_product(
                    model=prop, invert_matrix=True
                )
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Me{arg}I", MeI_prop)

        def MccDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MccProperty` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Mcc_{arg}_deriv"

            if getattr(self, stash_name, None) is None:
                M_prop_deriv = sp.diags(self.mesh.cell_volumes) * getattr(
                    self, f"{arg.lower()}Deriv"
                )
                setattr(self, stash_name, M_prop_deriv)
            return __inner_mat_mul_op(
                getattr(self, stash_name), u, v=v, adjoint=adjoint
            )

        setattr(cls, f"Mcc{arg}Deriv", MccDeriv_prop)

        def MnDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MnProperty` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Mn_{arg}_deriv"
            if getattr(self, stash_name, None) is None:
                M_prop_deriv = (
                    self.mesh.aveN2CC.T
                    * sp.diags(self.mesh.cell_volumes)
                    * getattr(self, f"{arg.lower()}Deriv")
                )
                setattr(self, stash_name, M_prop_deriv)
            return __inner_mat_mul_op(
                getattr(self, stash_name), u, v=v, adjoint=adjoint
            )

        setattr(cls, f"Mn{arg}Deriv", MnDeriv_prop)

        def MfDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MfProperty` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Mf_{arg}_deriv"
            if getattr(self, stash_name, None) is None:
                M_prop_deriv = self.mesh.get_face_inner_product_deriv(
                    np.ones(self.mesh.n_cells)
                )(np.ones(self.mesh.n_faces)) * getattr(self, f"{arg.lower()}Deriv")
                setattr(self, stash_name, M_prop_deriv)
            return __inner_mat_mul_op(
                getattr(self, stash_name), u, v=v, adjoint=adjoint
            )

        setattr(cls, f"Mf{arg}Deriv", MfDeriv_prop)

        def MeDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MeProperty` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Me_{arg}_deriv"
            if getattr(self, stash_name, None) is None:
                M_prop_deriv = self.mesh.get_edge_inner_product_deriv(
                    np.ones(self.mesh.n_cells)
                )(np.ones(self.mesh.n_edges)) * getattr(self, f"{arg.lower()}Deriv")
                setattr(self, stash_name, M_prop_deriv)
            return __inner_mat_mul_op(
                getattr(self, stash_name), u, v=v, adjoint=adjoint
            )

        setattr(cls, f"Me{arg}Deriv", MeDeriv_prop)

        def MccIDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MccPropertyI` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"Mcc{arg}I")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"Mcc{arg}Deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        setattr(cls, f"Mcc{arg}IDeriv", MccIDeriv_prop)

        def MnIDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MnPropertyI` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"Mn{arg}I")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"Mn{arg}Deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        setattr(cls, f"Mn{arg}IDeriv", MnIDeriv_prop)

        def MfIDeriv_prop(self, u, v=None, adjoint=False):
            """I
            Derivative of `MfPropertyI` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"Mf{arg}I")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"Mf{arg}Deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        setattr(cls, f"Mf{arg}IDeriv", MfIDeriv_prop)

        def MeIDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MePropertyI` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"Me{arg}I")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"Me{arg}Deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        setattr(cls, f"Me{arg}IDeriv", MeIDeriv_prop)

        @property
        def _clear_on_prop_update(self):
            items = [
                f"_Mcc_{arg}",
                f"_Mn_{arg}",
                f"_Mf_{arg}",
                f"_Me_{arg}",
                f"_MccI_{arg}",
                f"_MnI_{arg}",
                f"_MfI_{arg}",
                f"_MeI_{arg}",
                f"_Mcc_{arg}_deriv",
                f"_Mn_{arg}_deriv",
                f"_Mf_{arg}_deriv",
                f"_Me_{arg}_deriv",
            ]
            return items

        setattr(cls, f"_clear_on_{arg.lower()}_update", _clear_on_prop_update)
        return cls

    return decorator


class BasePDESimulation(BaseSimulation):
    @property
    def Vol(self):
        return self.Mcc

    @property
    def Mcc(self):
        """
        Cell center inner product matrix.
        """
        if getattr(self, "_Mcc", None) is None:
            self._Mcc = sp.diags(self.mesh.cell_volumes, format="csr")
        return self._Mcc

    @property
    def Mn(self):
        """
        Node inner product matrix.
        """
        if getattr(self, "_Mn", None) is None:
            vol = self.mesh.cell_volumes
            self._Mn = sp.diags(self.mesh.aveN2CC.T * vol, format="csr")
        return self._Mn

    @property
    def Mf(self):
        """
        Face inner product matrix.
        """
        if getattr(self, "_Mf", None) is None:
            self._Mf = self.mesh.get_face_inner_product()
        return self._Mf

    @property
    def Me(self):
        """
        Edge inner product matrix.
        """
        if getattr(self, "_Me", None) is None:
            self._Me = self.mesh.get_edge_inner_product()
        return self._Me

    @property
    def MccI(self):
        if getattr(self, "_MccI", None) is None:
            self._MccI = sp.diags(1.0 / self.mesh.cell_volumes, format="csr")
        return self._MccI

    @property
    def MnI(self):
        """
        Node inner product inverse matrix.
        """
        if getattr(self, "_MnI", None) is None:
            vol = self.mesh.cell_volumes
            self._MnI = sp.diags(1.0 / (self.mesh.aveN2CC.T * vol), format="csr")
        return self._MnI

    @property
    def MfI(self):
        """
        Face inner product inverse matrix.
        """
        if getattr(self, "_MfI", None) is None:
            self._MfI = self.mesh.get_face_inner_product(invert_matrix=True)
        return self._MfI

    @property
    def MeI(self):
        """
        Edge inner product inverse matrix.
        """
        if getattr(self, "_MeI", None) is None:
            self._MeI = self.mesh.get_edge_inner_product(invert_matrix=True)
        return self._MeI


@with_property_mass_matrices("sigma")
@with_property_mass_matrices("rho")
class BaseElectricalPDESimulation(BasePDESimulation):

    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")
    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")
    props.Reciprocal(sigma, rho)

    def __init__(
        self, mesh, sigma=None, sigmaMap=None, rho=None, rhoMap=None, **kwargs
    ):
        super().__init__(mesh=mesh, **kwargs)
        self.sigma = sigma
        self.rho = rho
        self.sigmaMap = sigmaMap
        self.rhoMap = rhoMap

    @property
    def deleteTheseOnModelUpdate(self):
        """
        matrices to be deleted if the model for conductivity/resistivity is updated
        """
        toDelete = super().deleteTheseOnModelUpdate
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete = (
                toDelete + self._clear_on_sigma_update + self._clear_on_rho_update
            )
        return toDelete

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ["sigma", "rho"]:
            for mat in self._clear_on_sigma_update + self._clear_on_rho_update:
                if hasattr(self, mat):
                    delattr(self, mat)


@with_property_mass_matrices("mu")
@with_property_mass_matrices("mui")
class BaseMagneticPDESimulation(BasePDESimulation):

    mu, muMap, muDeriv = props.Invertible(
        "Magnetic Permeability (H/m)",
    )
    mui, muiMap, muiDeriv = props.Invertible("Inverse Magnetic Permeability (m/H)")
    props.Reciprocal(mu, mui)

    def __init__(self, mesh, mu=mu_0, muMap=None, mui=None, muiMap=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.mu = mu
        self.mui = mui
        self.muMap = muMap
        self.muiMap = muiMap

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ["mu", "mui"]:
            for mat in self._clear_on_mu_update + self._clear_on_mui_update:
                if hasattr(self, mat):
                    delattr(self, mat)

    @property
    def deleteTheseOnModelUpdate(self):
        """
        items to be deleted if the model for Magnetic Permeability is updated
        """
        toDelete = super().deleteTheseOnModelUpdate
        if self.muMap is not None or self.muiMap is not None:
            toDelete = toDelete + self._clear_on_mu_update + self._clear_on_mui_update
        return toDelete
