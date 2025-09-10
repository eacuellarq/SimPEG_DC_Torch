
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import torch 

from matplotlib.colors import LogNorm,SymLogNorm
from discretize import TensorMesh
from discretize.utils import mkvc
from SimPEG.electromagnetics.static import resistivity as dc
from utils.results_inversion import SaveInversionProgress
from SimPEG.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage)

from SimPEG import (
    maps,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    
)

from SimPEG.config import SimpegConfig

class SimulationDCResistivity:
    def __init__(self, model=None, surveyinfo=None, adjust_model=False, m_background=100.0, tensor=False, protocol=None):
        self.tensor = tensor
        if tensor is True:
            self.model = model.flipud()
        else:
            self.model = model[::-1]
        
        self.surveyinfo = surveyinfo
        self.adjust_model = adjust_model
        self.return_matrix = False # If is false, return vector in all outputs
        self.protocol=protocol
        if isinstance(model, torch.Tensor):
            self.torch_is_active = True
        else:
            self.torch_is_active = False

    def initialize_survey(self):
        # Define survey
         # Posiciones de los electrodos (xi, xf)
        if self.protocol is None:

            nElec = self.surveyinfo["nElec"]
            sep = self.surveyinfo["sep"]

            pi = self.surveyinfo["pi"] # punto inicial 
            pf = sep*nElec # punto final
            self.end_locations = np.r_[self.surveyinfo["pi"], self.surveyinfo["pf"]]
            survey_type = self.surveyinfo["typ_survey"] # Tipo arreglo
            dimension_type = self.surveyinfo["dim"]
            data_type = self.surveyinfo["data_type"] # Tipo de datos: voltajes, resistividades, conductividades, apparent_resistivity

        

            station_separation = sep # Separación 
            num_rx_per_src = self.surveyinfo["nlines"] # Número máximo de líneas (n)

            # Definición de topografía en función del arreglo
            x = np.linspace(pi, pf, int(pf)) # el numero de elementos debe ser > a los elementos en el eje X de la malla
            z = np.zeros_like(x)
            # z = np.linspace(-pf/3, 0, int(pf/3)+1)
            x_topo, z_topo = np.meshgrid(x, z)
            topo_xz = np.c_[mkvc(x_topo), mkvc(z_topo)]

            topo_2d = np.unique(topo_xz[:, [0, 1]], axis=0)

            # Generate source list for DC survey line
            source_list = generate_dcip_sources_line(
                survey_type,
                data_type,
                dimension_type,
                self.end_locations,
                topo_2d,
                num_rx_per_src,
                station_separation,
            )

            self.survey = dc.survey.Survey(source_list, survey_type=survey_type)
            if self.surveyinfo["data_type"] == "apparent_resistivity":
                self.survey.set_geometric_factor()
        else:
            self.end_locations = np.r_[self.surveyinfo["pi"], self.surveyinfo["pf"]]
            a, b, m, n = self.protocol[:,0], self.protocol[:,1], self.protocol[:,2], self.protocol[:,3]
            nElec = self.protocol.max() + 1 
            # Number of the data
            ndata = a.size
            # ABMN locations
            a = np.c_[a, np.zeros(ndata)]
            b = np.c_[b, np.zeros(ndata)]
            m = np.c_[m, np.zeros(ndata)]
            n = np.c_[n, np.zeros(ndata)]

            
            self.IO = dc.IO()
            self.survey = self.IO.from_abmn_locations_to_survey(
                a,
                b,
                m,
                n,
                data_dc_type='apparent_resistivity',
                survey_type="dipole-dipole")
            if self.surveyinfo["data_type"] == "apparent_resistivity":
                self.survey.set_geometric_factor()


    # def initialize_mesh(self, custom_mesh=None, adjust_model=False, m_background=100.0):
    def initialize_mesh(self, custom_mesh=None, adjust_model=False, m_background=100.0, dx=None, dz=None):
        self.m_background = m_background
        
        l = np.diff(self.end_locations)[0]

            # l = np.diff(self.end_locations)[0]  # Compute total length
        self.dx = l / self.model.shape[1]  # Element width
        self.dz = self.surveyinfo["depth"] / self.model.shape[0]  # Element height


        self.adjust_model = adjust_model
        
        if adjust_model is True:
            hx = [(self.dx, self.model.shape[1])]
            hz = [(self.dz, self.model.shape[0])]
            mesh = TensorMesh([hx, hz], "0N")
            idx_model = None
            if self.tensor is True:
                m = self.model.log().flatten()
            else:
                m = np.log(self.model).flatten()
        else:
            if custom_mesh is None:
                self.n_pad_x = 12
                self.n_pad_z = 8
                pad_x = 1.2
                pad_z = 1.2
            else:
                self.n_pad_x = custom_mesh["n_pad_x"]
                self.n_pad_z = custom_mesh["n_pad_z"]
                pad_x = custom_mesh["pad_x"]
                pad_z = custom_mesh["pad_z"]
            hx = [(self.dx, self.n_pad_x, -pad_x), (self.dx, self.model.shape[1]), (self.dx, self.n_pad_x, pad_x)]
            hz = [(self.dz, self.n_pad_z, -pad_z), (self.dz, self.model.shape[0])]
            mesh = TensorMesh([hx, hz])
            mesh.origin = [-mesh.nodes_x[self.n_pad_x], -mesh.nodes_y[-1]]

            if self.tensor is True:
                m = (m_background * torch.ones(mesh.nC).to(SimpegConfig().device)).double()
                
                idx_model = ((torch.Tensor(mesh.gridCC[:,0]) > self.surveyinfo["pi"]) & 
                                (torch.Tensor(mesh.gridCC[:,0]) < self.surveyinfo["pi"] + l) &
                                (torch.Tensor(mesh.gridCC[:,1]) < 0.0) &
                                (torch.Tensor(mesh.gridCC[:,1]) > -self.surveyinfo["depth"])).to(SimpegConfig().device)

                m[idx_model] = self.model.flatten()
                
                m = m.log().flatten()

            else:
                m = m_background * np.ones(mesh.nC)
                idx_model = ((mesh.gridCC[:,0] > self.surveyinfo["pi"]) & 
                                (mesh.gridCC[:,0] < self.surveyinfo["pi"] + l) &
                                (mesh.gridCC[:,1] < 0.0) &
                                (mesh.gridCC[:,1] > -self.surveyinfo["depth"]))
                m[idx_model] = self.model.flatten()
                m = np.log(m).flatten()

        self.m = m
        self.idx_model = idx_model
        self.mapping = maps.ExpMap(mesh)
        self.mesh = mesh
        self.mx, self.mz = mesh.shape_cells

    def forward_modeling(self, nky=12, storeJ=True, add_noise=True, error = 0., only_data=False):
        self.nky=12
        self.simulation_dc = dc.Simulation2DNodal(self.mesh, rhoMap=self.mapping,
                                             survey=self.survey, storeJ=storeJ, nky=nky, torch_is_active=self.torch_is_active)

        self.dc_data = self.simulation_dc.make_synthetic_data(self.m, add_noise=add_noise, relative_error=error)

        if only_data is True:
            return apparent_resistivity_from_voltage(self.survey, self.dc_data.dobs)
        
        else:
            results_all = {'initialize': [self.survey, self.mesh], 
                        'simulation': [self.mapping, self.simulation_dc], 
                        'results': [self.dc_data, self.m, self.idx_model]}
            return results_all

    def Invert(self, floor=1e-3, percent_std=0.05, alpha_s=1e-3, 
            alpha_x=1, alpha_y=1, iter=20, b0_ratio=1e2, use_target=False,
            CF=2., CR=1):

        floor = floor
        percent_std = percent_std

        self.dc_data.standard_deviation = floor + percent_std * np.abs(self.dc_data.dobs)
        mref = np.log(np.median(apparent_resistivity_from_voltage(self.survey, self.dc_data.dobs)))*np.ones(self.mesh.nC)
        
        dmisfit = data_misfit.L2DataMisfit(data=self.dc_data, simulation=self.simulation_dc)
        reg = regularization.WeightedLeastSquares(
            self.mesh, alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, reference_model=mref
        )

        opt = optimization.InexactGaussNewton(maxIter=iter, maxIterCG=iter)
        opt.remember("xc")
        base_inv = inverse_problem.BaseInvProblem(dmisfit, reg, opt)

        beta_est = directives.BetaEstimate_ByEig(beta0_ratio=b0_ratio)
        target = directives.TargetMisfit(chifact=1)
        save = SaveInversionProgress()
        directives_list = [beta_est, save]

        if use_target is True:
            directives_list.append(target)

        beta_schedule = directives.BetaSchedule(coolingFactor=CF, coolingRate=CR)
        directives_list.append(beta_schedule)

        inv = inversion.BaseInversion(base_inv, directiveList=directives_list)
        model_recovered = inv.run(mref)
        inv_result = save.inversion_results

        inv_result["model_complete"] = copy.deepcopy(inv_result["model"])

        if self.adjust_model == False:

            for i in inv_result['iteration']:
                inv_result["model"][i-1][~self.idx_model] = np.log(self.m_background)

        self.inv_result = inv_result

        return inv_result 
    

    def error_inversion(self):
        iteration = self.inv_result["iteration"]
        error_model = []
        error_data = []

        for i in np.array(iteration)-1:
            a = np.exp(self.inv_result["model"][i][self.idx_model])
            b = np.exp(self.m[self.idx_model])
            error_model.append(np.sum(np.abs(a - b)) / np.sum(np.abs(b)) * 100)

            a = self.inv_result["dpred"][i]
            b = self.dc_data.dobs
            error_data.append(np.sum(np.abs(a - b)) / np.sum(np.abs(b)) * 100)

        self.error_model = np.vstack(error_model)
        self.error_data = np.vstack(error_data)

        return [self.error_data, self.error_model]

    def plot_model(
        self, m, ax=None, 
        colorbar=True,
        # xlim=[] 
    ):
        """
        A plotting function for our recovered model 
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        out = self.mesh.plot_image(
            self.mapping * m, pcolor_opts={'norm':LogNorm(), 'cmap':'jet'}, ax=ax,
            # clim=clim
        )

        ax.set_ylabel('Elevation (m)')
        ax.set_xlabel('Easting (m)')
        ax.set_aspect(1.5)  # some vertical exxageration
        
        if colorbar is True:
            cb = plt.colorbar(out[0], fraction=0.05, orientation='horizontal', ax=ax, pad=0.25)
            cb.set_label("Resistivity ($\Omega$m)")
        
        return ax

    def plot_tikhonov_curve(self, ax=None):
        """
        Plot the Tikhonov curve: Phi_d as a function of Phi_m
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        ax.plot(self.inv_result["phi_m"], self.inv_result["phi_d"], "-oC6", label="$\Phi_d$")
        ax.axhline(self.survey.nD/2, linestyle="--", color="k", label="$\Phi_d^*$")
        ax.axhline(self.survey.nD/2, linestyle="-.", color="k", label="$\chi\Phi_d^*$")
        ax.set_ylabel("$\Phi_d$")
        # ax.set_xlabel(x)
        
        ax.set_xlabel("$\Phi_m$")
        ax.set_ylabel("$\Phi_d$")
        
        ax.legend(loc=1)
        return ax

    def plot_normalized_misfit(self, iteration=None, ax=None):
        """
        Plot the normalized data misfit
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            
        if iteration is None:
            dpred = self.inv_result["dpred"][-1]
        else:
            dpred = self.inv_result["dpred"][iteration]
            
        normalized_misfit = (dpred - self.dc_data.dobs)/self.dc_data.standard_deviation
        out = dc.utils.plot_pseudosection(
            self.dc_data, dobs=normalized_misfit, data_type="misfit",
            plot_type="contourf", data_location=True, ax=ax, contourf_opts = {"cmap":'jet'},
            cbar_opts={"pad":0.25}
        )

        ax.set_title("normalized misfit")
        ax.set_yticklabels([])
        ax.set_aspect(1.5)  # some vertical exxageration
        ax.set_xlabel("Northing (m)")
        ax.set_ylabel("n-spacing")

        cb_axes = plt.gcf().get_axes()[-1]
        cb_axes.set_xlabel('normalized misfit')
        return ax

    def plot_error(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        iteration = self.inv_result["iteration"]
        idx_max = np.where(self.error_model.min() == self.error_model)[0][0]+1
        
        ax.plot(iteration, self.error_model, '-*C1', label='Error del modelo')
        ax.plot(iteration, self.error_data, '.-', label='Error de datos')
        ax.plot(idx_max, self.error_model.min(), 'oC2')
        ax.plot(idx_max, self.error_data.min(), 'oC2')
        ax.set_xlabel("Iter"); ax.set_ylabel("Error relativo %")
        ax.set_ylim(0,100)
        ax.legend()
        
        return ax
    def grad_plot(self, iteration=None,ax=None, colorbar=True,):

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if iteration is None:
            grad = self.inv_result["gradient"][-1]
        else:
            grad = self.inv_result["gradient"][iteration]
        
        out = self.mesh.plot_image(
            grad, pcolor_opts={'norm':SymLogNorm(0.001,1), 'cmap':'jet'}, ax=ax,
            # clim=clim
        )

        ax.set_ylabel('Elevation (m)')
        ax.set_xlabel('Easting (m)')
        ax.set_aspect(1.5)  # some vertical exxageration
        
        if colorbar is True:
            cb = plt.colorbar(out[0], fraction=0.05, orientation='horizontal', ax=ax, pad=0.25)
            cb.set_label("gradient [?]")
        
        return ax

    def plot_results_at_iteration(self, iteration=0):
        """
        A function to plot inversion results as a function of iteration 
        """
        fig = plt.figure(figsize=(12, 8))
        spec = gridspec.GridSpec(ncols=6, nrows=2, figure=fig)

        ax_tikhonov = fig.add_subplot(spec[0, :2])
        ax_error = fig.add_subplot(spec[1, :2])

        ax_grad = fig.add_subplot(spec[0, 2:])
        ax_model = fig.add_subplot(spec[1, 2:])

        self.plot_error(ax=ax_error)

        ax_error.plot(
            self.inv_result["iteration"][iteration], self.error_model[iteration], 
            'ks', ms=10
        )

        ax_error.plot(
            self.inv_result["iteration"][iteration], self.error_data[iteration], 
            'ks', ms=10
        )

        self.plot_tikhonov_curve(ax=ax_tikhonov)

        ax_tikhonov.plot(
            self.inv_result["phi_m"][iteration], self.inv_result["phi_d"][iteration], 
            'ks', ms=10
        )
        ax_tikhonov.set_title(f"iteration {iteration+1}")
        
        self.grad_plot(iteration=iteration, ax=ax_grad)

        self.plot_model(self.inv_result["model"][iteration], ax=ax_model, )
        # make the tikhonov plot square
        ax_tikhonov.set_aspect(1./ax_tikhonov.get_data_ratio())

        plt.tight_layout()

    def sensitivity(self, m=None, background=False, crop=False, transform='sum'):

        if m is None:
            m = self.m

        if background is True:
            m0 = np.log(self.m_background * np.ones(self.mesh.nC))
            f=self.simulation_dc.fields(m0)
            sens= self.simulation_dc._Jtvec(m0,f=f).sum(axis=1)
        
        elif transform == 'sum':
            f=self.simulation_dc.fields(m)
            sens= self.simulation_dc._Jtvec(m, f=f).sum(axis=1)
        elif transform is None:
            f=self.simulation_dc.fields(m)
            sens= self.simulation_dc._Jtvec(m, f=f)
        if self.return_matrix == True:
            mz, mx = self.mesh.shape_cells
            sens = sens.reshape(mz,mx)

            if crop == True:
                sens = sens[self.n_pad_z:, self.n_pad_x:-self.n_pad_x]
        
        return sens
    
    def grad(self, all=True):
        if all == True:
            dobs = apparent_resistivity_from_voltage(self.survey, volts=self.dc_data.dobs)
            grad=[]
            for i in range(len(self.inv_result["dpred"])):
                m = self.inv_result["model_complete"][i]
                J = self.sensitivity(m, transform=None)
                dpred = apparent_resistivity_from_voltage(self.survey, volts=self.inv_result["dpred"][i])
                grad.append(-(J @ (dpred-dobs)).flatten())
        self.inv_result["gradient"] = grad
        # print("Process successfully! Gradient add to inversion's dictionary")

    def fields(self, model=None, constrain_model=False):

        if model is None:
            model = self.m

        if constrain_model == True:
            hx = [(self.dx, self.model.shape[1])]
            hz = [(self.dz, self.model.shape[0])]
            mesh = TensorMesh([hx, hz], "0N")
            mapping = maps.ExpMap(mesh)
            simulation_dc = dc.Simulation2DNodal(mesh, rhoMap=mapping, solver=Solver, 
                                             survey=self.survey, storeJ=True, nky=self.nky)
            field = simulation_dc.fields(model)
            background_fields = simulation_dc.fields(np.log(self.m_background)*np.ones(mesh.nC))

        
        else:
            field = self.simulation_dc.fields(model)
            mesh = self.mesh
            simulation_dc_bg = dc.Simulation2DNodal(mesh, rhoMap=self.mapping, solver=Solver, 
                                             survey=self.survey, storeJ=True, nky=self.nky)

            background_fields = simulation_dc_bg.fields(np.log(self.m_background)*np.ones(mesh.nC))

        
        self.background_fields = background_fields

        self.meshField = mesh

        return field
    
    def charge_density(self, field, source_idx = 0, secundary = False):
        if source_idx == "all":
            c_d = field[self.survey.source_list, "charge_density"].sum(axis=1)  

        c_d = field[self.survey.source_list[source_idx], "charge_density"] 

        if secundary == True:
            c_d_background = self.background_fields[self.survey.source_list[source_idx], "charge_density"]
            c_d_secundary = c_d - c_d_background

            return c_d, c_d_secundary

        else:
            return c_d  

    def current_flow(self, field, source_idx = 0, secundary=False):
        if source_idx == "all":
            c_f = (self.meshField.aveE2CCV * field[self.survey.source_list, "j"]).sum(axis=1) 

        c_f = self.meshField.aveE2CCV * field[self.survey.source_list[source_idx], "j"]
        c_f = c_f.reshape(self.mz*2, self.mx)

        p1 = c_f[:self.mz, :]
        p2 = c_f[self.mz:, :]

        if secundary == True:
            c_d_background = (self.meshField.aveE2CCV *  self.background_fields[self.survey.source_list[source_idx], "j"]).reshape(self.mz*2, self.mx)
            c_f_secundary = c_f - c_d_background
            p1_secundary = c_f_secundary[:self.mz, :]
            p2_secundary = c_f_secundary[self.mz:, :]

            return c_f, p1, p2, c_f_secundary, p1_secundary, p2_secundary 
        
        else:
            return c_f, p1, p2

    def phi(self, field, source_idx = 0, secundary=False):
        if source_idx == "all":
            phi_f = (self.meshField.aveN2CC * field[self.survey.source_list, "phi"]).sum(axis=1) 

        phi_f = self.meshField.aveN2CC * field[self.survey.source_list[source_idx], "phi"]
        
        if secundary==True:
            phi_background = self.meshField.aveN2CC *  self.background_fields[self.survey.source_list[source_idx], "phi"]
            phi_secundary = phi_f - phi_background
            return phi_f, phi_secundary
        else:
            return phi_f

    def e(self, field, source_idx = 0, secundary=False):
        if source_idx == "all":
            ef = (self.meshField.aveE2CCV * field[self.survey.source_list, "e"]).sum(axis=1) 

        ef = self.meshField.aveE2CCV * field[self.survey.source_list[source_idx], "e"]

        ef = ef.reshape(self.mz*2, self.mx)
        p1 = ef[:self.mz, :]
        p2 = ef[self.mz:, :]   

        if secundary == True:
            ef_background = (self.meshField.aveE2CCV *  self.background_fields[self.survey.source_list[source_idx], "e"]).reshape(self.mz*2, self.mx)
            ef_secundary = ef - ef_background
            p1_secundary = ef_secundary[:self.mz, :]
            p2_secundary = ef_secundary[self.mz:, :]

            return ef, p1, p2, ef_secundary, p1_secundary, p2_secundary

        else:
            return ef, p1, p2 
        
    def vector_fields(self, v=None, mesh=None, range_x=None, range_y=None):
        if mesh is None:
            mesh = self.mesh

        v = v.flatten()
        U, V  = mesh.reshape(v.reshape((mesh.nC,-1), order='F'), "CC", "CC", "M")

        hxmin = mesh.h[0].min()
        hymin = mesh.h[1].min()

        if range_x is not None:
            dx = range_x[1] - range_x[0]
            nxi = int(dx / hxmin)
            hx = np.ones(nxi) * dx / nxi
            origin_x = range_x[0]
        else:
            nxi = int(mesh.h[0].sum()/hxmin)
            hx = np.ones(nxi) * mesh.h[0].sum() / nxi
            origin_x = mesh.origin[0]
        if range_y is not None:
            dy = range_y[1] - range_y[0]
            nyi = int(dy / hymin)
            hy = np.ones(nyi) * dy / nyi
            origin_y = range_y[0]
        else:
            nyi = int(mesh.h[1].sum()/hymin)
            hy = np.ones(nyi) * mesh.h[1].sum() / nyi
            origin_y = mesh.origin[1]

        tMi = mesh.__class__(h=[hx,hy], origin=np.r_[origin_x, origin_y])
        P = mesh.get_interpolation_matrix(tMi.gridCC, "CC", zeros_outside=True)
        Ui = tMi.reshape(P * mkvc(U), "CC", "CC", "M")
        Vi = tMi.reshape(P * mkvc(V), "CC", "CC", "M")

        return [tMi.cell_centers_x, tMi.cell_centers_y, Ui.T, Vi.T]