import argparse
from pathlib import Path
from textwrap import wrap

import torch
from PIL import ImageDraw, ImageFont
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from image2text.config import Config
from image2text.data import END_IDX, MAX_CAPTION_LENGTH, PAD_IDX, START_IDX, Flickr, prepare_data
from image2text.image_captioner import ImageCaptioner


def save_comparison(image, prediction, ground_truth, output_path):
    # Undo Flickr's ImageNet normalization before rendering.
    mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    display_image = to_pil_image((image * std + mean).clamp(0, 1)).resize((384, 384))

    draw = ImageDraw.Draw(display_image)
    font = ImageFont.load_default()
    captions = "\n".join(
        "\n".join(wrap(f"{label}: {caption}", width=42))
        for label, caption in (("Prediction", prediction), ("Ground truth", ground_truth))
    )
    _, top, _, bottom = draw.multiline_textbbox((0, 0), captions, font=font, spacing=4)
    draw.rectangle((0, 0, display_image.width, bottom - top + 16), fill="black")
    draw.multiline_text((8, 8 - top), captions, fill="white", font=font, spacing=4)
    display_image.save(output_path)

def main(split="train", decode="sample", encoder="custom"):
    config = Config.for_encoder(encoder)
    checkpoint = config.best_checkpoint
    if encoder == "custom" and not checkpoint.is_file():
        checkpoint = Path("checkpoints/model.pth")
    weights = torch.load(checkpoint, weights_only=True, map_location="cpu")
    train_image_paths, train_image_captions, test_image_paths, test_image_captions, vocab = prepare_data(
        legacy_vocab=checkpoint != config.best_checkpoint
    )
    if split == "train":
        dataset = Flickr(train_image_paths, train_image_captions, vocab, image_size=config.image_size)
    else:
        dataset = Flickr(test_image_paths, test_image_captions, vocab, image_size=config.image_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptioner(config=config, num_emb=len(vocab))
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    words_by_id = {index: word for word, index in vocab.items()}

    def caption_text(token_ids):
        words = []
        for token_id in token_ids:
            if token_id == END_IDX:
                break
            if token_id not in (START_IDX, PAD_IDX):
                words.append(words_by_id[token_id])
        return " ".join(words)

    @torch.inference_mode()
    def caption(image):
        input_tokens = torch.tensor([[START_IDX]], device=device)
        image_embedding = model.vision_encoder(image.to(device))
        for _ in range(MAX_CAPTION_LENGTH):
            logits = model.vision_decoder(input_tokens, image_embedding)[:, -1, :]
            if decode == "greedy":
                next_token = logits.argmax(dim=-1)
            else:
                next_token = Categorical(logits=logits).sample()
            if next_token.item() == END_IDX:
                break
            input_tokens = torch.cat((input_tokens, next_token.unsqueeze(1)), dim=1)
        return caption_text(input_tokens[0].tolist())

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    images, _, target_captions, _ = next(iter(loader))
    for index, (image, target) in enumerate(zip(images, target_captions)):
        prediction = caption(image.unsqueeze(0))
        ground_truth = caption_text(target.tolist())
        save_comparison(image, prediction, ground_truth, results_dir / f"comparison_{index}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Flickr8k caption comparisons.")
    parser.add_argument("--split", choices=("train", "test"), default="train",
                        help="Dataset split to sample (default: train)")
    parser.add_argument("--decode", choices=("sample", "greedy"), default="sample",
                        help="Sample each word or choose the highest-scoring word (default: sample)")
    parser.add_argument("--encoder", choices=("custom", "dinov2"), default="custom")
    args = parser.parse_args()
    main(args.split, args.decode, args.encoder)
