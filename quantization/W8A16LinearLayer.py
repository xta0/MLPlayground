import torch
import torch.nn as nn
import torch.nn.functional as F


class W8A16LinearLayer(nn.Module):
    def __init__(self, 
                in_features,
                out_features, 
                bias=True, 
                dtype=torch.float32):
        super().__init__()

        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128, 127, 
                (out_features, in_features), 
                dtype=torch.int8
            )
        )
        
        self.register_buffer(
            "scales", 
            torch.randn((out_features), dtype=dtype)
        )
        
        if bias:
            self.register_buffer(
                "bias", 
                torch.randn((1, out_features), dtype=dtype)
            )
        else:
            self.bias = None

    def quantize(self, weights):
        # w_fp32: [m, n]
        w_fp32 = weights.clone().to(torch.float32)
        # returns the max value per row
        # scales: [m]
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)
        # scales: [m, 1]
        scales = scales.unsqueeze(1)

        # apply per-channel linear quantization
        w_fp32 = torch.round(weights/scales).to(torch.int8)

        self.int8_weights = w_fp32
        self.scales = scales


                
    def forward(self, input):
        return w8_a16_forward(self.int8_weights, 
                              input, self.scales, self.bias)

def w8_a16_forward(weight, input, scales, bias=None):
    
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales
    
    if bias is not None:
        output = output + bias
      
    return output

def replace_linear_with_target(module, 
                            target_class, 
                            module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias

            new_module = target_class(child.in_features, 
                                    child.out_features, 
                                    old_bias is not None, 
                                    child.weight.dtype)
            setattr(module, name, new_module)
            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target(
                child, target_class, module_name_to_exclude)

def main():
    module = W8A16LinearLayer(4, 8)
    print("Weights before:\n" , module.int8_weights)
    random_matrix = torch.randn((4, 8), dtype=torch.bfloat16)
    module.quantize(random_matrix)
    print("Weights After:\n" , module.int8_weights)
    print("Weights.shape:\n" , module.int8_weights.shape)
    print("scale: ", module.scales)
    print("scale.shape: ", module.scales.shape)

    # ### dequantized weights
    dq_w = module.int8_weights * module.scales.unsqueeze(1)
    print(dq_w)
    print(random_matrix)

    print("mean: ", (random_matrix - dq_w).abs().mean())



if __name__ == "__main__":
    main()
