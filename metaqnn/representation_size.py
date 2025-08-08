from models.layers import AbstractLayer

# Representation size bins according to Baker et al. (2017)
# R_SIZE_BINS = {
#     'large': float('inf'),  # [infinite, 8]
#     'medium': 8,           # (8, 4]
#     'small': 4             # (4, 1]
# }


def get_representation_size_bin(r_size: float) -> str:
    """Bin representation sizes into discrete buckets"""
    if r_size > 8:
        return 'large'
    elif r_size > 4:
        return 'medium'
    else:
        return 'small'


def calculate_new_representation_size(current_r_size: float, layer: AbstractLayer) -> float:
    """Calculate new representation size after applying a layer"""
    if layer.type == 'C':
        # Assuming same padding, convolution doesn't change size
        return current_r_size
    elif layer.type == 'P':
        # Pooling reduces size by stride
        return current_r_size / layer.stride
    elif layer.type in ['FC', 'GAP', 'SM']:
        return 1.0
    return current_r_size
