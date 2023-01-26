from math import e


def error(output):
    return output * (1 - output)

def sigmoid(weighted_sum, gamma=1):
    return 1 / (1 + e ** (-gamma * weighted_sum))


if __name__ == '__main__':
    # model
    input_layer = [0., 0.]
    in_weights = [[1., 1.], [1., 1.]]

    layer_1 = [0., 0.]
    l1_weights = [[1., 1.], [1., 1.]]

    layer_2 = [0., 0.]
    l2_weights = [[1.], [1.]]

    out_layer = [0.]

    print(input_layer, in_weights, layer_1, l1_weights, layer_2, l2_weights, out_layer)

    # forward prop
    value = [0., 1.]

    input_layer = value
    layer_1 = [sigmoid(input_layer[0] * in_weights[0][0] + input_layer[1] * in_weights[0][1]), 
               sigmoid(input_layer[1] * in_weights[1][0] + input_layer[1] * in_weights[1][1])]

    layer_2 = [sigmoid(layer_1[0] * l1_weights[0][0] + layer_1[1] * l1_weights[0][1]),
               sigmoid(layer_1[1] * l1_weights[1][0] + layer_1[1] * l1_weights[1][1])]
    
    out_layer = [sigmoid(layer_2[0] * l2_weights[0][0] + layer_2[1] * l2_weights[1][0])]

    print(input_layer, in_weights, layer_1, l1_weights, layer_2, l2_weights, out_layer)

    # backprop
    target = [1.]
    l2_error_term = (target[0] - out_layer[0]) * error(out_layer[0])
    l2_delta = l2_error_term * out_layer[0]
    l2_weights = [[l2_weights[0][0] - l2_delta], [l2_weights[1][0] - l2_delta]]
    l1_delta = []
   
    print(l2_weights)