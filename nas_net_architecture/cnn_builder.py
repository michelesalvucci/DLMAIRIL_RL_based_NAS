from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, AveragePooling2D, SeparableConv2D, Add, Concatenate

from parameters import IMG_SIZE, MOTIF_REPETITIONS


def build_child_network(cell_structure, reduction_cell_structure, num_cells=MOTIF_REPETITIONS, input_shape=(IMG_SIZE,IMG_SIZE,1), filters=32):
    """
    Build a child network by stacking normal and reduction cells.
    - cell_structure: architecture of the normal cell
    - reduction_cell_structure: architecture of the reduction cell
    - num_cells: number of cells to stack
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    h_i, h_i_1 = x, x
    for i in range(num_cells):
        if i % 3 == 2:  # Insert reduction cell every 3 cells
            x = __build_cell(reduction_cell_structure, [h_i, h_i_1], filters, reduction=True)
        else:
            x = __build_cell(cell_structure, [h_i, h_i_1], filters)
        h_i_1, h_i = h_i, x
        filters *= 2 if i % 3 == 2 else 1  # Double filters after reduction
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    model = Model(inputs=inputs, outputs=outputs)
    return model


def __build_cell(cell_structure, inputs, filters, reduction=False):
    # Start with initial hidden states (inputs)
    outputs = [inputs[0], inputs[1]]
    # For each block, create a new hidden state and append it to outputs
    for block in cell_structure:
        # For reduction cell, first op uses stride 2
        stride = 2 if reduction and len(outputs) < 2 else 1
        # Select input2 index based on controller prediction (could be h_i_1 or previous block output)
        inp2_idx = 1 if len(outputs) > 1 else 0
        inp2 = outputs[inp2_idx] if block["input2"] == "h_i_1" else outputs[0]
        op1_out = __apply_operation(outputs[0], block["op1"], filters, stride)
        op2_out = __apply_operation(inp2, block["op2"], filters, stride)
        if block["combine"] == "element-wise-addition":
            # Ensure channel dimensions match for Add()
            ch1 = op1_out.shape[-1]
            ch2 = op2_out.shape[-1]
            if ch1 != ch2:
                # Project op2_out to match op1_out's channels
                op2_out = Conv2D(ch1, (1,1), padding='same', activation=None)(op2_out)
            out = Add()([op1_out, op2_out])
        else:
            out = Concatenate()([op1_out, op2_out])
        # Append the newly-created hidden state so it can be used as input in subsequent blocks
        outputs.append(out)
    # Concatenate unused outputs as final cell output (as per controller prediction)
    unused_outputs = [outputs[i+2] for i, block in enumerate(cell_structure) if block["output"] in block.get("concatenated_unused_outputs",[])]
    if unused_outputs:
        cell_output = Concatenate()(unused_outputs)
    else:
        cell_output = outputs[-1]
    return cell_output


def __apply_operation(x, op_name, filters, stride=1):
    # Map operation names to TensorFlow layers
    if op_name == "identity":
        return x
    elif op_name == "1x3 then 3x1 convolution":
        x = Conv2D(filters, (1,3), strides=stride, padding='same', activation='relu')(x)
        return Conv2D(filters, (3,1), strides=1, padding='same', activation='relu')(x)
    elif op_name == "1x7 then 7x1 convolution":
        x = Conv2D(filters, (1,7), strides=stride, padding='same', activation='relu')(x)
        return Conv2D(filters, (7,1), strides=1, padding='same', activation='relu')(x)
    elif op_name == "3x3 dilated convolution":
        return Conv2D(filters, (3,3), strides=stride, padding='same', activation='relu', dilation_rate=2)(x)
    elif op_name == "3x3 average pooling":
        return AveragePooling2D(pool_size=(3,3), strides=stride, padding='same')(x)
    elif op_name == "3x3 max pooling":
        return MaxPooling2D(pool_size=(3,3), strides=stride, padding='same')(x)
    elif op_name == "5x5 max pooling":
        return MaxPooling2D(pool_size=(5,5), strides=stride, padding='same')(x)
    elif op_name == "7x7 max pooling":
        return MaxPooling2D(pool_size=(7,7), strides=stride, padding='same')(x)
    elif op_name == "1x1 convolution":
        return Conv2D(filters, (1,1), strides=stride, padding='same', activation='relu')(x)
    elif op_name == "3x3 convolution":
        return Conv2D(filters, (3,3), strides=stride, padding='same', activation='relu')(x)
    elif op_name == "3x3 depthwise-separable conv":
        x = SeparableConv2D(filters, (3,3), strides=stride, padding='same', activation='relu')(x)
        return x
    elif op_name == "5x5 depthwise-seperable conv":
        x = SeparableConv2D(filters, (5,5), strides=stride, padding='same', activation='relu')(x)
        return x
    elif op_name == "7x7 depthwise-separable conv":
        x = SeparableConv2D(filters, (7,7), strides=stride, padding='same', activation='relu')(x)
        return x
    else:
        raise ValueError(f"Unknown operation: {op_name}")