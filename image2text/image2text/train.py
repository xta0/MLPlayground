import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from image2text.data import prepare_data
from image2text.data import Flickr
from image2text.utils import extract_patches
from image2text.image_captioner import ImageCaptioner
from image2text.data import PAD_TOKEN
from image2text.config import BEST_CHECKPOINT, Config
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tempfile import TemporaryDirectory

def train(model, train_loader, val_loader, device, pad_id, epochs=50, best_checkpoint=BEST_CHECKPOINT):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    best_val_loss = float("inf")

    def batch_loss(images, input_tokens, target_tokens):
        images = images.to(device)
        tokens_in = input_tokens.to(device, dtype=torch.long)
        target_ids = target_tokens.to(device, dtype=torch.long)

        input_attention_mask = tokens_in.ne(pad_id)
        token_count = target_ids.ne(pad_id).sum().item()
        if token_count == 0:
            return None, 0

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            """
            ImageCaptioner.forward() returns logits of shape [B, T, V], where B is the batch size, 
            T is the number of caption input positions, and V = len(vocab). 
            The decoder's fc_out maps each position's hidden vector from hidden_size to V scores, 
            one for each possible next token.

            There's no softmax after fc_out because nn.CrossEntropyLoss expects raw logits and computes log-softmax internally. 
            The training loop transposes the logits to [B, V, T] so the vocabulary is on the class axis, 
            then compares them with target token IDs of shape [B, T].
            
            For one caption, <start> a dog runs, the output shape is [1, 4, V]. 
            The four predictions live along the T axis. 
            Each position has a vector of V scores, one for every token in the vocabulary.

            | Logit vector | Input prefix it sees | Correct next token |
            |---|---|---|
            | logits[0, 0, :] | <start> | a |
            | logits[0, 1, :] | <start> a | dog |
            | logits[0, 2, :] | <start> a dog | runs |
            | logits[0, 3, :] | <start> a dog runs | <end> |
            """
            logits = model(images, tokens_in, input_attention_mask)
            loss_sum = loss_fn(logits.transpose(1, 2), target_ids)

        return loss_sum, token_count
    with SummaryWriter() as writer:
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_tokens = 0

            for images, inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                optimizer.zero_grad(set_to_none=True)
                loss_sum, token_count = batch_loss(images, inputs, targets)
                if token_count == 0:
                    continue

                loss = loss_sum / token_count
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss_sum.detach().item()
                train_tokens += token_count

            if train_tokens == 0:
                raise ValueError("The training loader produced no non-padding targets.")

            message = f"Epoch {epoch}: train loss {train_loss_sum / train_tokens:.4f}"
            writer.add_scalar("loss/train", train_loss_sum / train_tokens, epoch)

            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_tokens = 0

                with torch.no_grad():
                    for images, inputs, targets, _ in val_loader:
                        loss_sum, token_count = batch_loss(images, inputs, targets)
                        if token_count == 0:
                            continue

                        val_loss_sum += loss_sum.item()
                        val_tokens += token_count

                if val_tokens == 0:
                    raise ValueError("The validation loader produced no non-padding targets.")
                val_loss = val_loss_sum / val_tokens
                writer.add_scalar("loss/val", val_loss, epoch)
                message += f", validation loss {val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    with TemporaryDirectory(dir=best_checkpoint.parent) as directory:
                        temporary = Path(directory) / best_checkpoint.name
                        torch.save(model.state_dict(), temporary)
                        temporary.replace(best_checkpoint)
                    best_val_loss = val_loss
                    message += " (saved best checkpoint)"

            print(message)

    if val_loader is None:
        checkpoint = Path("checkpoints/model.pth")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)

def main(encoder="custom"):
    config = Config.for_encoder(encoder)
    train_image_paths, train_image_captions, val_image_paths, val_image_captions, vocab = prepare_data(
        evaluation_split="val"
    )
    print("Train image paths:", len(train_image_paths))
    print("Train image captions:", len(train_image_captions))
    print("Validation images:", len(val_image_paths))
    print("Vocab size:", len(vocab))
    train_set = Flickr(train_image_paths, train_image_captions, vocab, image_size=config.image_size)
    train_loader = DataLoader(train_set, 
                              batch_size=128, 
                              shuffle=True)
    val_set = Flickr(val_image_paths, val_image_captions, vocab, image_size=config.image_size)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on device:", device)
    model = ImageCaptioner(
        config=config,
        num_emb=len(vocab)
    ).to(device)
    num_model_params = 0
    for param in model.parameters():
        num_model_params += param.numel()
    print("Number of model parameters:", num_model_params) # 9545219 ~10 million
    train(model, train_loader, val_loader, device, pad_id=vocab[PAD_TOKEN], best_checkpoint=config.best_checkpoint)


def _inspect_sample(sample):
    print(type(sample))
    print(len(sample))
    test_images, test_tokens, test_targets, test_mask = sample
    print(test_images[0].shape) # [3, 128, 128]
    print(test_tokens[0].shape) # [50]
    print(len(test_targets)) # 128
    print(len(test_mask)) # 128

    image = test_images[0].unsqueeze(0) # Add batch dimension
    patches = extract_patches(image)
    print(patches.shape) # torch.Size([1, 256, 192])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Flickr8k image captioner.")
    parser.add_argument("--encoder", choices=("custom", "dinov2"), default="custom")
    main(parser.parse_args().encoder)
