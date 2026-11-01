import torch
from safetensors.torch import load_file

def load_lora_weights(pipeline, lora_path, alpha = 0.5, device = 'cpu'):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    state_dict = load_file(lora_path)
    visited = []
    for key in state_dict:
        if '.alpha' in key or key in visited:
            continue
        if 'text' in key:
            layer_infos = key.split('.')[0].split(
                LORA_PREFIX_TEXT_ENCODER + "_"
            )[-1].split('_')
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split('.')[0].split(
                LORA_PREFIX_UNET + "_"
            )[-1].split('_')
            curr_layer = pipeline.unet

        # loop through the layers to find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                # no exception means the layer is found
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                # all names are pop put, break out from the loop
                elif len(layer_infos) == 0:
                    break
            except Exception:
                # exception means the layer is not found, try the next name
                if len(temp_name) > 0:
                    temp_name += "_"+layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        
        # updating the checkpoint model weights
        # snsure the sequence of lora_up(A) then lora_down(B)
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else: 
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
        
        
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up   = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up   = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
        
        print(curr_layer.weight.data.shape)

        # update visited list, ensure no duplicated weight is processed
        for item in pair_keys:
            visited.append(item)
