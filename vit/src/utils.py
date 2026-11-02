import json, os
import torch
import matplotlib.pyplot as plt
import math
from data import CLASSES

def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)


class EarlyStop:
    def __init__(self, patience=3):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.counter = 0
        self.min_loss = float('inf')

    def stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        should_stop = self.counter >= self.patience
        return should_stop


def visualize_attention_maps(attention_maps, ground_truth_images, labels, predictions, output_path="attention_maps.png"):
    """Save original images beside attention maps, with true and predicted classes."""
    print(attention_maps[0].shape)
    with torch.no_grad():
        attention_maps = torch.cat(attention_maps, dim=1)
        print(f"attention map shape: {attention_maps.shape}")
        # Focuses on the attention between the CLP token and the 64 image tokens
        attention_maps = attention_maps[:, :, 0, 1:]
        print(f"attention map shape: {attention_maps.shape}")
        attention_maps = attention_maps.mean(dim=1)
        print(f"attention map shape: {attention_maps.shape}")
        num_patches = attention_maps.size(-1)
        size = int(math.sqrt(num_patches))
        attention_maps = attention_maps.view(-1, size, size)
        print(f"attention map shape: {attention_maps.shape}")
        attention_maps = attention_maps.unsqueeze(1)
        attention_maps = torch.nn.functional.interpolate(attention_maps, size=(32, 32),
                                                         mode='bilinear', align_corners=False)
        attention_maps = attention_maps.squeeze(1)
        print(f"attention map shape: {attention_maps.shape}")
    count = attention_maps.size(0)
    if count == 0:
        raise ValueError("Cannot visualize an empty batch")
    if not (len(ground_truth_images) == len(labels) == len(predictions) == count):
        raise ValueError("Images, labels, predictions, and attention maps must have matching batch sizes")
    # Undo the evaluation transform's normalization: (pixel - 0.5) / 0.5.
    images = (ground_truth_images.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
    labels = labels.detach().cpu().tolist()
    predictions = predictions.detach().cpu().tolist()
    pairs_per_row = min(2, count)
    rows = math.ceil(count / pairs_per_row)
    fig = plt.figure(figsize=(pairs_per_row * 5, rows * 3), dpi=100)
    try:
        panels = fig.subfigures(rows, pairs_per_row, squeeze=False)
        for i, panel in enumerate(panels.flat):
            if i >= count:
                continue
            panel.suptitle(
                f"label: {CLASSES[labels[i]]}/ pred: {CLASSES[predictions[i]]}",
                color="red" if labels[i] != predictions[i] else "black",
            )
            image_ax, attention_ax = panel.subplots(1, 2)
            image_ax.imshow(images[i].permute(1, 2, 0))
            image_ax.set_title("Original")
            attention_ax.imshow(attention_maps[i].detach().float().cpu(), cmap="jet")
            attention_ax.set_title("Attention")
            image_ax.axis("off")
            attention_ax.axis("off")
            panel.subplots_adjust(top=0.78, left=0.04, right=0.96, wspace=0.08)
        os.makedirs(os.path.dirname(os.fspath(output_path)) or ".", exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved attention grid to {output_path}")
    return output_path
 
