import numpy as np


# input/output channel being 1
def conv2d(
    tensor,
    kernel,
):
    n = tensor.shape[0]
    f = kernel.shape[1]
    output_shape = (n - f + 1, n - f + 1)
    output = np.zeros(output_shape)
    for i in range(output_shape[0]):  #loop height
        for j in range(output_shape[0]):  #loop width
            tmp = 0
            for k in range(f):
                for l in range(f):
                    kv = kernel[k][l]
                    tv = tensor[i + k][j + l]
                    tmp += kv * tv
            output[i][j] = tmp

    return output


input = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])

kernel = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

print(conv2d(input, kernel))