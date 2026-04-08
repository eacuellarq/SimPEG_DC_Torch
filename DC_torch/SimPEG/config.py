import torch

class SimpegConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._torch_is_active = False
            cls._instance._device = 'cpu'
            cls._instance._dtype = torch.float64  # valor por defecto
            cls._instance._solver = 'superlu'      # nuevo atributo por defecto
        return cls._instance

    @property
    def torch_is_active(self):
        return self._torch_is_active

    @torch_is_active.setter
    def torch_is_active(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("torch_is_active must be a boolean")
        self._torch_is_active = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value: str):
        if not isinstance(value, str):
            raise TypeError("device must be a string")
        self._device = value

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if not isinstance(value, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        self._dtype = value

    @property
    def solver(self):
        """Devuelve el solver en uso: 'superlu', 'pardiso' o 'pcg_gpu'."""
        return self._solver

    @solver.setter
    def solver(self, value: str):
        if not isinstance(value, str):
            raise TypeError("solver must be a string")
        value_lower = value.lower()
        if value_lower not in ('superlu', 'pardiso', 'pcg_gpu', 'dense_gpu'):
            raise ValueError("solver must be 'superlu', 'pardiso', 'pcg_gpu', or 'dense_gpu'")
        self._solver = value_lower
