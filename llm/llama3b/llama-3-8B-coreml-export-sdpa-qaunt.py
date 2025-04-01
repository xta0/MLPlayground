# pip install transformers==4.44.2
# pip install coremltools numpy

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
import coremltools as ct
import numpy as np

import sentencepiece

# https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM

class BaselineLlamaForCausalLM(LlamaForCausalLM):
    """Baseline LlamaForCausalLM model without key/value caching."""
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        out = super().forward(
            input_ids,
            attention_mask,
            use_cache=False,
        )
        return out.logits

model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
torch_model = BaselineLlamaForCausalLM.from_pretrained(model_id).eval()

# Count the attention blocks
attention_blocks = [module for module in torch_model.modules() if "attention" in str(type(module)).lower()]
print(f"Number of attention blocks: {len(attention_blocks)}")

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)

prompt = "What is GenAI?"
inputs = tokenizer(prompt, return_tensors='pt')

# Extract input_ids and attention_mask
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# convert to coreml

batch_size, context_size = 1, 1024
input_shape = (batch_size, context_size)

# trace the PyTorch model
example_inputs: tuple[torch.Tensor] = (
    torch.zeros(input_shape, dtype=torch.int32),
    torch.zeros(input_shape, dtype=torch.int32),
)
traced_model: torch.jit.ScriptModule = torch.jit.trace(
    torch_model,
    example_inputs=example_inputs,
)

# convert to Core ML format
inputs: list[ct.TensorType] = [
    ct.TensorType(shape=input_shape, dtype=np.int32, name="inputIds"),
    ct.TensorType(shape=input_shape, dtype=np.int32, name="attentionMask"),
]

outputs: list[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
mlmodel: ct.models.MLModel = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=outputs,
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)

# quantization
# op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
#     mode="linear_symmetric",
#     dtype="int4",
#     granularity="per_block",
#     block_size=32,
# )
# config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
# mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(
#     mlmodel, config=config
# )

mlmodel.save("Llama-3.1-8B-Instruct.mlpackage")  # 3.5GB, int4
