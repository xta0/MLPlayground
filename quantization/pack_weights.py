import torch

def pack_weights(uint8tensor, bits):
    if uint8tensor.shape[0] * bits % 8 != 0:
        raise ValueError(f"The input shape needs to be a mutiple \
        of {8 / bits} - got {uint8tensor.shape[0]}")

    num_values = uint8tensor.shape[0] * bits // 8

    num_steps = 8 // bits

    unpacked_idx = 0

    packed_tensor = torch.zeros((num_values), dtype=torch.uint8)

    # 1 0 3 2 - 01 00 11 10

    # [0000 0000] -> 0000 0001

    # 0000 0001

    # 0000 0000 - 0000 0000

    # 0000 0011 - 0011 0000 - 0011 0001

    # 1011 0001
    
    for i in range(num_values):
        for j in range(num_steps):
            packed_tensor[i] |= uint8tensor[unpacked_idx] << (bits * j)
            unpacked_idx += 1
    return packed_tensor


def main():
    unpacked_tensor = torch.tensor([1, 0, 3, 2], 
                               dtype=torch.uint8)
    pw1 = pack_weights(unpacked_tensor, 2)
    print(pw1)
    unpacked_tensor = torch.tensor([1, 0, 3, 2, 3, 3, 3, 3], 
                               dtype=torch.uint8)
    pw2 = pack_weights(unpacked_tensor, 2)
    print(pw2)

if __name__ == "__main__":
    main()
