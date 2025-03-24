import torch
from linear_symmetric import get_q_scale_symmetric, linear_q_with_scale_and_zero_point

def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):
    output_dim = r_tensor.shape[dim]
    # store the scales
    scale = torch.zeros(output_dim)

    for index in range(output_dim):
        sub_tensor = r_tensor.select(dim, index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype=dtype)

    # reshape the scale
    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_q_with_scale_and_zero_point(
        r_tensor, scale=scale, zero_point=0, dtype=dtype)
   
    return quantized_tensor, scale


def main():
    test_tensor=torch.tensor(
        [[191.6, -13.5, 728.6],
        [92.14, 295.5,  -184]]
    )
    print(test_tensor.shape)
    print(test_tensor)
    print("--------------------")
    # random values for scale and zero point
    q_tensor, scale = linear_q_symmetric_per_channel(test_tensor, 0)
    print(q_tensor)
    print(scale)
    print(scale.shape)
    print("--------------------")
    q_tensor, scale = linear_q_symmetric_per_channel(test_tensor, 1)
    print(q_tensor)
    print(scale)
    print(scale.shape)


if __name__ == "__main__":
    main()

