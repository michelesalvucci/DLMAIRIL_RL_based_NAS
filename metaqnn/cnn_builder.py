from typing import List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten, Dropout, Input

from models.layers import AbstractLayer, ConvolutionalLayer, PoolingLayer, FullyConnectedLayer, SoftmaxLayer, GlobalAveragePoolingLayer
from child_trainer import IMG_SIZE


def build_cnn(architecture: List[AbstractLayer]) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
    
    """ Calculate total number of dropout layers (every 2 layers, excluding SoftmaxLayer) """
    non_softmax_layers = [l for l in architecture if not isinstance(l, SoftmaxLayer)]
    total_dropout_layers = len(non_softmax_layers) // 2
    dropout_count = 0
    
    """ Generate the architecture """
    for i, layer in enumerate(architecture):
        if isinstance(layer, ConvolutionalLayer):
            conv_layer = Conv2D(
                filters=layer.receptive_fields,
                kernel_size=layer.field_size,
                kernel_initializer='glorot_uniform',
                strides=layer.stride,
                padding='same' if i > 0 else 'valid',
                activation='relu'
            )
            model.add(conv_layer)

        elif isinstance(layer, PoolingLayer):
            pooling_layer = MaxPooling2D(
                pool_size=layer.field_size,
                strides=layer.stride,
                padding='same'
            )
            model.add(pooling_layer)

        elif isinstance(layer, FullyConnectedLayer):
            """Flatten before first FC layer if coming from conv layers"""
            if i > 0 and not isinstance(architecture[i-1], FullyConnectedLayer):
                model.add(Flatten())
            fc_layer = Dense(
                units=layer.neurons,
                activation='relu'
            )
            model.add(fc_layer)

        elif isinstance(layer, GlobalAveragePoolingLayer):
            gap_layer = GlobalAveragePooling2D()
            model.add(gap_layer)

            model.add(Dense(
                units=1,
                activation='sigmoid'
            ))

        elif isinstance(layer, SoftmaxLayer):
            """Add final classification layer"""
            if len([l for l in architecture if isinstance(l, FullyConnectedLayer)]) == 0:
                """If no FC layers, add flatten before output"""
                model.add(Flatten())
            output_layer = Dense(
                units=1,
                activation='sigmoid'
            )
            model.add(output_layer)

        """ Add dropout layer after every 2 layers, excluding SoftmaxLayer - Baker et al. (2017, p. 6) """
        if not isinstance(layer, SoftmaxLayer) and (i + 1) % 2 == 0:
            dropout_count += 1
            dropout_prob = dropout_count / (2 * total_dropout_layers) if total_dropout_layers > 0 else 0
            model.add(Dropout(dropout_prob))
    
    return model
