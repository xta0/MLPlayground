# pip install qai_hub_models

from qai_hub_models.models.ffnet_40s import Model as FFNet_40s
import torch

# Load from pre-trained weights
ffnet_40s = FFNet_40s.from_pretrained()
input_shape = (1, 3, 1024, 2048)
example_inputs = torch.rand(input_shape)
traced_model = torch.jit.trace(ffnet_40s, example_inputs)
print(traced_model)

# psnr > 30 db