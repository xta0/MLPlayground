import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from diffusion_utilities import *

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  #assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)        # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[10, 256, 4,  4]
        
         # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample 
            nn.GroupNorm(8, 2 * n_feat), # normalize                        
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat), # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)       #[10, 256, 8, 8]
        down2 = self.down2(down1)   #[10, 256, 4, 4]
        
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)
        
        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
            
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)     # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        #print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")


        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out
# --------------------------
# Setup: device and hyperparameters
# --------------------------
train_on_mps = torch.backends.mps.is_available()
if train_on_mps:
    print('MPS is available.  Training on MPS ...')
    device = torch.device("mps")
else:
    print ("MPS device not found.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_feat = 64
n_cfeat = 5
height = 16   # image is 16x16
timesteps = 500

# --------------------------
# Construct the DDPM noise schedule
# --------------------------
beta1 = 1e-4
beta2 = 0.02
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
# Compute cumulative product in log space to avoid numerical issues.
ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
ab_t[0] = 1  # ensure the first element is 1

# =============================================================================
# Helper Function: Denoise and Add Noise Back
# =============================================================================
def denoise_add_noise(x, t, pred_noise, z=None):
    """
    Given current sample x at timestep t, predicted noise pred_noise,
    and (optionally) additional noise z, compute the updated x.
    """
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

# =============================================================================
# Sampling Function for DDPM
# =============================================================================
@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    """
    Reverse diffusion process.
    n_sample: number of images to generate.
    save_rate: frequency for storing intermediate samples.
    Returns:
      - samples: the final generated images (tensor)
      - intermediate: a NumPy array containing intermediate samples.
    """
    samples = torch.randn(n_sample, 3, height, height).to(device)
    intermediate = []  # to store intermediate steps for visualization

    # Reverse diffusion: from timestep 'timesteps' down to 1
    for i in range(timesteps, 0, -1):
        print(f"Sampling timestep {i:3d}", end="\r")
        # Create a time tensor with shape [1, 1, 1, 1] for broadcasting
        t_tensor = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        # For timesteps greater than 1, add noise z; for the final step, set z=0.
        z = torch.randn_like(samples) if i > 1 else 0
        # Predict noise using the network
        eps = nn_model(samples, t_tensor)
        # Update the sample with the denoising function
        samples = denoise_add_noise(samples, i, eps, z)
        # Save intermediate samples if desired
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

# =============================================================================
# Visualization Helper: Display the Final Samples in a Grid
# =============================================================================
def visualize_samples(samples, nrow=8):
    """
    Arrange generated samples into a grid and display them.
    """
    grid_img = make_grid(samples, nrow=nrow, padding=2)
    plt.figure(figsize=(8, 8))
    # Rearrange dimensions to (height, width, channels) for imshow
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.title("DDPM Generated Samples")
    plt.axis("off")
    plt.show()

def animate_sample_process(intermediate, interval=200, save=False, save_path="animation.gif"):
    """
    Animate the reverse diffusion process for the first image in the batch.
    
    Parameters:
      intermediate : NumPy array of shape (num_frames, batch, channels, height, width)
      interval     : Delay between frames in milliseconds
      save         : If True, save the animation as a GIF to save_path
      save_path    : Path for saving the GIF if save is True
    
    Returns:
      ani          : The FuncAnimation object.
    """
    num_frames = intermediate.shape[0]
    fig, ax = plt.subplots()
    
    # Use the first sample of the batch for animation; transpose to (H, W, C)
    img = np.transpose(intermediate[0, 0], (1, 2, 0))
    im = ax.imshow(img, animated=True)
    ax.axis("off")
    
    def update(frame):
        img = np.transpose(intermediate[frame, 0], (1, 2, 0))
        im.set_array(img)
        ax.set_title(f"Step {frame}")
        return [im]
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True, repeat=True)
    
    if save:
        ani.save(save_path, writer=PillowWriter(fps=1000 / interval))
    
    plt.show()
    return ani
# =============================================================================
# Animation Helper: Animate All Samples in One Grid
# =============================================================================
def animate_all_samples(intermediate, nrow=8, interval=200, save=False, save_path="animation.gif"):
    """
    Animate the reverse diffusion process for all samples in the batch.
    
    Parameters:
      intermediate : NumPy array of shape (num_frames, n_sample, channels, height, width)
      nrow         : number of images per row in the grid.
      interval     : Delay between frames in milliseconds.
      save         : If True, save the animation as a GIF to save_path.
      save_path    : Path for saving the GIF if save is True.
      
    Returns:
      ani          : The FuncAnimation object.
    """
    num_frames = intermediate.shape[0]
    fig, ax = plt.subplots()
    
    # For the first frame, convert the batch (32 samples) into a grid.
    frame_tensor = torch.tensor(intermediate[0])
    grid_img = make_grid(frame_tensor, nrow=nrow, padding=2)
    grid_img_np = grid_img.permute(1, 2, 0).numpy()
    
    im = ax.imshow(grid_img_np, animated=True)
    ax.axis("off")
    
    def update(frame):
        frame_tensor = torch.tensor(intermediate[frame])
        grid_img = make_grid(frame_tensor, nrow=nrow, padding=2)
        grid_img_np = grid_img.permute(1, 2, 0).numpy()
        im.set_array(grid_img_np)
        ax.set_title(f"Step {frame}")
        return [im]
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True, repeat=True)
    
    if save:
        ani.save(save_path, writer=PillowWriter(fps=1000 / interval))
    
    plt.show()
    return ani

# =============================================================================
# Main Function
# =============================================================================
def main():
    global nn_model  # so that sample_ddpm can use nn_model
    model_path = "./weights/model_trained.pth"  # path to your pre-trained model

    # Instantiate the model with the same hyperparameters as during training
    nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    # Load pre-trained weights and set the model to evaluation mode
    nn_model.load_state_dict(torch.load(model_path, map_location=device))
    nn_model.eval()
    print("Pre-trained model loaded successfully.")

    # Run the sampling process to generate images
    n_sample = 32  # number of images to generate
    samples, intermediate_ddpm = sample_ddpm(n_sample)
    
    # Visualize the final generated samples
    # visualize_samples(samples, nrow=8)
    # Animate the sampling process for one image (the first sample).
    print("Animating the sampling process for one image...")
    # animate_sample_process(intermediate_ddpm, interval=200, save=False)
    animate_all_samples(intermediate_ddpm, nrow=8, interval=200, save=True)

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    main()