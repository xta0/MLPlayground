import torch
from linear_symmetric import linear_quantization

def quantized_linear_W8A32_without_bias(input, q_w, s_w, z_w):
    assert input.dtype == torch.float32
    assert q_w.dtype == torch.int8

    dequantized_weight = s_w * (q_w.to(torch.float32)  - z_w)
    output = torch.nn.functional.linear(input, dequantized_weight)
    
    return output

def main():
    x = torch.tensor([[1, 2, 3]], dtype=torch.float32) # [1,3]
    weight = torch.rand(3, 3) #[3, 3]
    q_w,s_w = linear_quantization(weight)
    y = quantized_linear_W8A32_without_bias(x, q_w, s_w, 0)
    print(y)

if __name__ == "__main__":
    main()
