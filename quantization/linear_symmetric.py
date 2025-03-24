import torch

def get_q_scale_symmetric(tensor, dtype=torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max

    # return the scale
    return r_max/q_max

def linear_q_with_scale_and_zero_point(
    tensor, scale, zero_point, dtype = torch.int8):

    scaled_and_shifted_tensor = tensor / scale + zero_point

    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min,q_max).to(dtype)
    
    return q_tensor

def linear_quantization(tensor, dtype=torch.int8):
    scale = get_q_scale_symmetric(tensor, dtype=dtype)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor,
                                                          scale, 
                                                          0, 
                                                          dtype=dtype)
    
    return quantized_tensor, scale

def main():
    test_tensor=torch.tensor(
        [[191.6, -13.5, 728.6],
        [92.14, 295.5,  -184],
        [0,     684.6, 245.5]]
    )
    # random values for scale and zero point
    q_tensor, scale = linear_quantization(test_tensor)
    print(q_tensor)
    print(scale)

if __name__ == "__main__":
    main()