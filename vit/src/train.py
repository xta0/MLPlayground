import torch
from torch import nn, optim
from data import prepare_data
from vit import ViTForClassification
from utils import EarlyStop, save_experiment, save_checkpoint
from torch.utils.tensorboard import SummaryWriter

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, exp_name):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.exp_name = exp_name
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train(self, train_loader, test_loader, epochs, save_model_every_n_epochs=0,
              patience=3):
        train_losses, test_losses, accuracies = [], [], []
        early_stop = EarlyStop(patience=patience)
        with SummaryWriter(log_dir=f"runs/{self.exp_name}") as writer:
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader)
                accuracy, test_loss = self.evaluate(test_loader)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                accuracies.append(accuracy)
                writer.add_scalar("Loss/train", train_loss, epoch + 1)
                writer.add_scalar("Loss/test", test_loss, epoch + 1)
                writer.add_scalar("Accuracy/test", accuracy, epoch + 1)
                writer.flush()
                print(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
                if test_loss < early_stop.min_loss:
                    save_checkpoint(self.exp_name, self.model, "best")
                should_stop = early_stop.stop(test_loss)
                if save_model_every_n_epochs > 0 and (epoch + 1) % save_model_every_n_epochs == 0:
                    print('\tSave checkpoint at epoch', epoch+1)
                    save_checkpoint(self.exp_name, self.model, epoch+1)

                if should_stop:
                    print(f"Early stopping at epoch {epoch+1}: loss did not improve "
                          f"for {patience} epochs.")
                    break

        # Save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)


    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_images = 0
        for images, labels in train_loader:
            # images: [B, 3, 32, 32]
            # labels: [B]
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast( device_type=self.device.type, enabled=self.use_amp):
                # model outpus (logits, all_attn_probs)
                logits, _ = self.model(images) # [B, 10]
                loss = self.loss_fn(logits, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * len(images)
            total_images += len(images)
        return total_loss / total_images

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_images = 0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits, _ = self.model(images) # [B, 10]
                    loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)
                # Exactly. For each image, argmax(dim=1) selects the index of its highest score among the 10 classes.
                prediction = torch.argmax(logits, dim=1) # [B] 
                correct += (prediction == labels).sum().item()
                total_images += len(images)
        accuracy = correct / total_images
        avg_loss = total_loss / total_images
        return accuracy, avg_loss

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after this many epochs without evaluation loss improvement")

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args

def main():
    args = parse_args()
    # Training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    trainloader, testloader, classes = prepare_data(batch_size)

        # Example: Iterate through the training data
    for images, labels in trainloader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break  # Just to demonstrate, we break after the first batch

    model = ViTForClassification(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, device, exp_name=args.exp_name)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs,
                  patience=args.patience)


if __name__ == "__main__":
    main()
    
    
    
