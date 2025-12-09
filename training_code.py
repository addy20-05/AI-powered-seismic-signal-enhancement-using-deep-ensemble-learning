# training_code.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ================================
# ----- MATPLOTLIB STYLE FOR GOOGLE COLAB
# ================================


# Only use universally supported rcParams in Colab
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# ================================
# ----- DATA LOADING FROM sample.csv
# ================================


print("Loading data from sample.csv...")


# Read the CSV file
df = pd.read_csv('sample.csv')


# Display basic info about the data
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Trace categories: {df['trace_category'].unique()}")


# Convert trace_data from string to numpy arrays
trace_data = np.stack([np.array(eval(s)) for s in df['trace_data']])
trace_categories = df['trace_category'].values


print(f"Trace data shape: {trace_data.shape}")
print(f"Sample length: {trace_data.shape[1]}")


# Convert labels to numerical values
label_mapping = {'earthquake': 0, 'noise': 1, 'blast': 2}
labels = np.array([label_mapping[lab] for lab in trace_categories])


print(f"Label distribution:")
for label_name, label_id in label_mapping.items():
   count = np.sum(labels == label_id)
   print(f"  {label_name}: {count} samples")


# Split data
X_train, X_test, y_train, y_test = train_test_split(
   trace_data, labels, stratify=labels, test_size=0.3, random_state=42
)


# Normalize data
X_train = (X_train - X_train.mean()) / (X_train.std() + 1e-8)
X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)


# Convert to PyTorch tensors
X_train_torch = torch.from_numpy(X_train).float().unsqueeze(1)  # (batch, 1, length)
X_test_torch = torch.from_numpy(X_test).float().unsqueeze(1)


print(f"Training data: {X_train_torch.shape}")
print(f"Test data: {X_test_torch.shape}")


# Save test data for later use
torch.save({
   'X_test': X_test_torch,
   'y_test': y_test,
   'X_train': X_train_torch,
   'y_train': y_train
}, 'test_data.pth')


print("Test data saved to 'test_data.pth'")


# ================================
# ----- DENOISER MODEL DEFINITIONS
# ================================


class DnCNN(nn.Module):
   def __init__(self, channels=1, num_layers=8):
       super(DnCNN, self).__init__()
       layers = []
       layers.append(nn.Conv1d(channels, 64, kernel_size=3, padding=1))
       layers.append(nn.ReLU(inplace=True))

       for _ in range(num_layers - 2):
           layers.append(nn.Conv1d(64, 64, kernel_size=3, padding=1))
           layers.append(nn.BatchNorm1d(64))
           layers.append(nn.ReLU(inplace=True))

       layers.append(nn.Conv1d(64, channels, kernel_size=3, padding=1))
       self.dncnn = nn.Sequential(*layers)

   def forward(self, x):
       out = self.dncnn(x)
       return x - out  # Residual learning


class UNet1D(nn.Module):
   def __init__(self, in_channels=1, out_channels=1):
       super(UNet1D, self).__init__()

       # Encoder
       self.enc1 = self._block(in_channels, 64)
       self.enc2 = self._block(64, 128)
       self.enc3 = self._block(128, 256)

       # Decoder
       self.dec3 = self._block(256, 128)
       self.dec2 = self._block(128, 64)
       self.dec1 = nn.Conv1d(64, out_channels, kernel_size=3, padding=1)

       self.pool = nn.MaxPool1d(2)
       self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

   def _block(self, in_channels, out_channels):
       return nn.Sequential(
           nn.Conv1d(in_channels, out_channels, 3, padding=1),
           nn.BatchNorm1d(out_channels),
           nn.ReLU(inplace=True),
           nn.Conv1d(out_channels, out_channels, 3, padding=1),
           nn.BatchNorm1d(out_channels),
           nn.ReLU(inplace=True)
       )

   def forward(self, x):
       # Encoder
       e1 = self.enc1(x)
       e2 = self.enc2(self.pool(e1))
       e3 = self.enc3(self.pool(e2))

       # Decoder
       d3 = self.dec3(self.upsample(e3))
       d2 = self.dec2(self.upsample(d3 + e2[:, :, :d3.shape[2]]))
       d1 = self.dec1(self.upsample(d2 + e1[:, :, :d2.shape[2]]))

       return x - d1[:, :, :x.shape[2]]  # Residual learning


class Noise2Noise(nn.Module):
   def __init__(self, channels=1):
       super(Noise2Noise, self).__init__()
       self.net = nn.Sequential(
           nn.Conv1d(channels, 64, 3, padding=1),
           nn.ReLU(inplace=True),
           nn.Conv1d(64, 64, 3, padding=1),
           nn.ReLU(inplace=True),
           nn.Conv1d(64, 64, 3, padding=1),
           nn.ReLU(inplace=True),
           nn.Conv1d(64, 32, 3, padding=1),
           nn.ReLU(inplace=True),
           nn.Conv1d(32, channels, 3, padding=1)
       )

   def forward(self, x):
       return self.net(x)


class MLPDenoiser(nn.Module):
   def __init__(self, input_size=1000, hidden_size=512):
       super(MLPDenoiser, self).__init__()
       self.net = nn.Sequential(
           nn.Linear(input_size, hidden_size),
           nn.ReLU(inplace=True),
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU(inplace=True),
           nn.Linear(hidden_size, hidden_size//2),
           nn.ReLU(inplace=True),
           nn.Linear(hidden_size//2, input_size)
       )

   def forward(self, x):
       batch_size, channels, length = x.shape
       x_flat = x.view(batch_size, -1)
       denoised = self.net(x_flat)
       return denoised.view(batch_size, channels, length)


# ===========================
# ----- TRAINING DENOISERS
# ===========================


def train_denoiser(model, train_data, epochs=10000, model_name=""):
   """Train a denoising model"""
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   optimizer = optim.Adam(model.parameters(), lr=1e-3)
   criterion = nn.MSELoss()

   # For Noise2Noise, we need pairs of noisy samples
   if model_name == "n2n":
       # Create another noisy version of the same clean signal
       noisy_data_2 = train_data + 0.1 * torch.randn_like(train_data)
       train_data_2 = noisy_data_2.to(device)
   else:
       train_data_2 = train_data.to(device)

   train_data = train_data.to(device)

   model.train()
   losses = []
   for epoch in range(epochs):
       optimizer.zero_grad()
       output = model(train_data)

       if model_name == "n2n":
           loss = criterion(output, train_data_2)
       else:
           # Self-supervised denoising - denoise to a slightly less noisy version
           target = train_data + 0.05 * torch.randn_like(train_data)
           loss = criterion(output, target)

       loss.backward()
       optimizer.step()
       losses.append(loss.item())

       if (epoch + 1) % 5 == 0:
           print(f'{model_name} - Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

   return model, losses


print("\nTraining denoising models...")


# Get the actual signal length from data
signal_length = X_train_torch.shape[2]
print(f"Signal length: {signal_length}")


# Initialize models with correct dimensions
dncnn = DnCNN()
unet = UNet1D()
n2n = Noise2Noise()
mlp_denoiser = MLPDenoiser(input_size=signal_length)


# Train individual models
print("Training DnCNN...")
dncnn, dncnn_losses = train_denoiser(dncnn, X_train_torch, epochs=500, model_name="DnCNN")


print("Training U-Net...")
unet, unet_losses = train_denoiser(unet, X_train_torch, epochs=500, model_name="U-Net")


print("Training Noise2Noise...")
n2n, n2n_losses = train_denoiser(n2n, X_train_torch, epochs=500, model_name="N2N")


print("Training MLP Denoiser...")
mlp_denoiser, mlp_losses = train_denoiser(mlp_denoiser, X_train_torch, epochs=500, model_name="MLP")


# ===========================
# ----- SAVE TRAINED MODELS
# ===========================


print("\nSaving trained models...")


# Save individual models
torch.save(dncnn.state_dict(), 'dncnn_denoiser.pth')
torch.save(unet.state_dict(), 'unet_denoiser.pth')
torch.save(n2n.state_dict(), 'n2n_denoiser.pth')
torch.save(mlp_denoiser.state_dict(), 'mlp_denoiser.pth')


# Save training losses for plotting
training_losses = {
   'dncnn': dncnn_losses,
   'unet': unet_losses,
   'n2n': n2n_losses,
   'mlp': mlp_losses
}
torch.save(training_losses, 'training_losses.pth')


print("All models saved successfully!")
print("Saved files:")
print("- dncnn_denoiser.pth")
print("- unet_denoiser.pth")
print("- n2n_denoiser.pth")
print("- mlp_denoiser.pth")
print("- test_data.pth")
print("- training_losses.pth")


# ===========================
# ----- PLOT WITH BOLD AXES (COLAB COMPATIBLE)
# ===========================


def create_bold_plot():
   """Create training loss plot with bold axes and values for Colab"""
   fig, ax = plt.subplots(figsize=(12, 8))

   # Plot lines with thicker style
   ax.plot(dncnn_losses, label='DnCNN', linewidth=2.5, color='blue')
   ax.plot(unet_losses, label='U-Net', linewidth=2.5, color='red')
   ax.plot(n2n_losses, label='N2N', linewidth=2.5, color='green')
   ax.plot(mlp_losses, label='MLP', linewidth=2.5, color='magenta')

   # Set labels with bold font using direct method
   ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
   ax.set_ylabel('Loss', fontsize=16, fontweight='bold')
   ax.set_title('Training Losses for Denoising Models', fontsize=18, fontweight='bold')

   # Make legend bold using prop parameter
   ax.legend(prop={'weight': 'bold', 'size': 14}, framealpha=0.9)

   # Grid with higher visibility
   ax.grid(True, alpha=0.4, linewidth=1.2)

   # Make axis spines (lines) thicker
   for spine in ax.spines.values():
       spine.set_linewidth(2.5)

   # Set tick parameters
   ax.tick_params(axis='both', which='major', labelsize=14, width=2.5, length=6)

   # Manually set bold font for tick labels
   for label in ax.get_xticklabels():
       label.set_fontweight('bold')
   for label in ax.get_yticklabels():
       label.set_fontweight('bold')

   plt.tight_layout()
   return fig, ax


# Create and display the plot
print("\nGenerating training loss plot with bold axes...")
fig, ax = create_bold_plot()
plt.show()


# Save the plot
fig.savefig('training_losses_bold.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Plot saved as 'training_losses_bold.png'")


print("\n=== TRAINING COMPLETE ===")
print("Now run the testing code separately!")
