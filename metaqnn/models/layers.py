from typing import Tuple


class AbstractLayer():
    layer_depth: int
    type: str

    def __init__(self, layer_depth: int):
        self.layer_depth = layer_depth
    

class ConvolutionalLayer(AbstractLayer):
    type: str = 'C'
    receptive_fields: int
    field_size: int
    stride: int
    
    def __init__(self, layer_depth: int, receptive_fields: int, field_size: int, stride: int):
        super().__init__(layer_depth)
        self.receptive_fields = receptive_fields
        self.field_size = field_size
        self.stride = stride

    def __str__(self):
        return (f"type={self.type}, "
                f"depth={self.layer_depth}, "
                f"receptive_fields={self.receptive_fields}, "
                f"field_size={self.field_size}, "
                f"stride={self.stride}")
    
    def short_description(self) -> str:
        """Return a short description of the layer for logging"""
        return f"({self.receptive_fields},{self.field_size},{self.stride})"
        

class PoolingLayer(AbstractLayer):
    type: str = 'P'
    field_size: int
    stride: int
    
    def __init__(self, layer_depth: int, field_size: int, stride: int):
        super().__init__(layer_depth)
        self.field_size = field_size
        self.stride = stride

    def __str__(self):
        return (f"type={self.type}, "
                f"depth={self.layer_depth}, "
                f"field_size={self.field_size}, "
                f"stride={self.stride}")
    
    def short_description(self) -> str:
        """Return a short description of the layer for logging"""
        return f"({self.field_size},{self.stride})"


class FullyConnectedLayer(AbstractLayer):
    type: str = 'FC'
    neurons: int
    
    def __init__(self, layer_depth: int, neurons: int):
        super().__init__(layer_depth)
        self.neurons = neurons

    def __str__(self):
        return (f"type={self.type}, "
                f"depth={self.layer_depth}, "
                f"neurons={self.neurons}")
    
    def short_description(self) -> str:
        """Return a short description of the layer for logging"""
        return f"({self.neurons})"


class GlobalAveragePoolingLayer(AbstractLayer):
    type: str = 'GAP'
    
    def __init__(self, layer_depth: int):
        super().__init__(layer_depth)

    def __str__(self):
        return (f"type={self.type}, "
                f"depth={self.layer_depth}")

    def short_description(self) -> str:
        """Return a short description of the layer for logging"""
        return f"()"


class SoftmaxLayer(AbstractLayer):
    type: str = 'SM'
    
    def __init__(self, layer_depth: int):
        super().__init__(layer_depth)

    def __str__(self):
        return (f"type={self.type}, "
                f"depth={self.layer_depth}")
    
    def short_description(self) -> str:
        """Return a short description of the layer for logging"""
        return f"()"
