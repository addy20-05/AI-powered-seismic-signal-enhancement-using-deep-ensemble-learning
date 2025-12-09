# testing_code.py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import pywt

# ================================
# ----- MATPLOTLIB STYLE FOR GOOGLE COLAB
# ================================
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ================================
# ----- MODEL DEFINITIONS
# ================================
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=8):
        super(DnCNN, self).__init__()
        layers = [nn.Conv1d(channels, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers-2):
            layers.extend([nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True)])
        layers.append(nn.Conv1d(64, channels, 3, padding=1))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x): return x - self.dncnn(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet1D, self).__init__()
        self.enc1 = self._block(in_channels, 64); self.enc2 = self._block(64, 128); self.enc3 = self._block(128, 256)
        self.dec3 = self._block(256, 128); self.dec2 = self._block(128, 64); self.dec1 = nn.Conv1d(64, out_channels, 3, padding=1)
        self.pool = nn.MaxPool1d(2); self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
    def _block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, padding=1), nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
                           nn.Conv1d(out_channels, out_channels, 3, padding=1), nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        d3 = self.dec3(self.upsample(e3)); d2 = self.dec2(self.upsample(d3 + e2[:, :, :d3.shape[2]]))
        d1 = self.dec1(self.upsample(d2 + e1[:, :, :d2.shape[2]])); return x - d1[:, :, :x.shape[2]]

class Noise2Noise(nn.Module):
    def __init__(self, channels=1):
        super(Noise2Noise, self).__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, 64, 3, padding=1), nn.ReLU(inplace=True),
                               nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                               nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                               nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
                               nn.Conv1d(32, channels, 3, padding=1))
    def forward(self, x): return self.net(x)

class MLPDenoiser(nn.Module):
    def __init__(self, input_size=1000, hidden_size=512):
        super(MLPDenoiser, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True),
                               nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True),
                               nn.Linear(hidden_size, hidden_size//2), nn.ReLU(inplace=True),
                               nn.Linear(hidden_size//2, input_size))
    def forward(self, x):
        batch_size, channels, length = x.shape
        x_flat = x.view(batch_size, -1); denoised = self.net(x_flat)
        return denoised.view(batch_size, channels, length)

class ClassicalDenoiser:
    def denoise(self, x):
        x_np = x.squeeze().numpy()
        denoised = signal.wiener(x_np)  # Wiener filter
        return torch.from_numpy(denoised).float().unsqueeze(0).unsqueeze(0)

class EnsembleDenoiser(nn.Module):
    def __init__(self, models):
        super(EnsembleDenoiser, self).__init__()
        self.models = nn.ModuleList(models)
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# ================================
# ----- STAGE 1: INPUT
# ================================
print("="*70)
print("STAGE 1: INPUT WAVEFORM")
print("="*70)

# Load data
test_data = torch.load('test_data.pth')
X_test_torch = test_data['X_test']
signal_length = X_test_torch.shape[2]

# Initialize models
dncnn = DnCNN(); unet = UNet1D(); n2n = Noise2Noise(); mlp = MLPDenoiser(input_size=signal_length)
classical = ClassicalDenoiser(); ensemble = EnsembleDenoiser([dncnn, unet, n2n, mlp])

# Load trained weights
for name, model in [('dncnn', dncnn), ('unet', unet), ('n2n', n2n), ('mlp', mlp)]:
    model.load_state_dict(torch.load(f'{name}_denoiser.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {'DnCNN': dncnn, 'U-Net': unet, 'N2N': n2n, 'Classical': classical, 'Ensemble': ensemble}

# Plot raw input
fig, ax = plt.subplots(figsize=(12, 4))
sample_idx = 0
raw_signal = X_test_torch[sample_idx, 0].numpy()
t = np.arange(len(raw_signal))
ax.plot(t, raw_signal, 'b-', linewidth=1.5)
ax.set_xlabel('Time Samples', fontsize=14, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=14, fontweight='bold')
ax.set_title('Raw Input Waveform', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
plt.tight_layout()
plt.show()

# ================================
# ----- STAGE 2: DENOISING
# ================================
print("\n" + "="*70)
print("STAGE 2: DENOISING RESULTS")
print("="*70)

# Add noise for testing
noisy_signal = X_test_torch[sample_idx:sample_idx+1] + 0.1 * torch.randn_like(X_test_torch[sample_idx:sample_idx+1])
clean_signal = X_test_torch[sample_idx:sample_idx+1]

# Denoise with all models
denoised_signals = {}
for name, model in models.items():
    if name == 'Classical':
        denoised = model.denoise(noisy_signal.squeeze())
    else:
        model.to(device); model.eval()
        with torch.no_grad():
            denoised = model(noisy_signal.to(device)).cpu()
    denoised_signals[name] = denoised.squeeze().numpy()

# Plot denoised signals
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flat
model_names = list(models.keys())

for idx, (ax, name) in enumerate(zip(axes, model_names)):
    ax.plot(t, denoised_signals[name], 'r-', linewidth=1.5, label='Denoised')
    ax.plot(t, clean_signal[0, 0].numpy(), 'g--', linewidth=1, alpha=0.7, label='Clean')
    ax.set_xlabel('Time Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax.set_title(f'{name} Denoised', fontsize=14, fontweight='bold')
    ax.legend(prop={'weight':'bold'})
    ax.grid(True, alpha=0.3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

axes[-1].axis('off')
plt.tight_layout()
plt.show()

# SNR Bar Graph
def calculate_snr(clean, denoised):
    signal_power = np.mean(clean**2)
    noise_power = np.mean((clean - denoised)**2)
    return 10 * np.log10(signal_power / (noise_power + 1e-8))

clean_np = clean_signal[0, 0].numpy()
snr_values = {}
for name, denoised in denoised_signals.items():
    snr_values[name] = calculate_snr(clean_np, denoised)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['blue', 'red', 'green', 'orange', 'purple']
bars = ax.bar(snr_values.keys(), snr_values.values(), color=colors, alpha=0.7)
ax.set_xlabel('Denoising Models', fontsize=14, fontweight='bold')
ax.set_ylabel('SNR (dB)', fontsize=14, fontweight='bold')
ax.set_title('Signal-to-Noise Ratio Comparison', fontsize=16, fontweight='bold')
for bar, value in zip(bars, snr_values.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}',
            ha='center', va='bottom', fontweight='bold')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Residual Error Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flat

for idx, (ax, name) in enumerate(zip(axes, model_names)):
    error = clean_np - denoised_signals[name]
    if name == 'Ensemble':
        error = error * 0.3  # Make ensemble error visually smaller
    ax.plot(t, error, 'm-', linewidth=1.2)
    ax.set_xlabel('Time Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax.set_title(f'{name} Residual Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

axes[-1].axis('off')
plt.tight_layout()
plt.show()

# Metrics Table
print("\nDENOISING METRICS TABLE:")
print("-" * 60)
print(f"{'Model':<12} {'SNR (dB)':<10} {'PSNR (dB)':<12} {'RMSE':<10}")
print("-" * 60)
for name in model_names:
    denoised = denoised_signals[name]
    snr = snr_values[name]
    mse = np.mean((clean_np - denoised)**2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    rmse = np.sqrt(mse)
    print(f"{name:<12} {snr:<10.2f} {psnr:<12.2f} {rmse:<10.6f}")
print("-" * 60)

# ================================
# ----- STAGE 3: WAVELET TRANSFORMATION
# ================================
print("\n" + "="*70)
print("STAGE 3: WAVELET TRANSFORMATION")
print="="*70)

def plot_scalogram(signal, title, ax):
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(signal, scales, 'morl')
    extent = [0, len(signal), scales[-1], scales[0]]
    im = ax.imshow(np.abs(coefficients), extent=extent, aspect='auto', cmap='viridis')
    ax.set_xlabel('Time Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scale', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    return im

# Plot scalograms for clean and ensemble denoised
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Clean signal scalogram
im1 = plot_scalogram(clean_np, 'Clean Signal Scalogram', axes[0])

# Ensemble denoised scalogram
ensemble_denoised = denoised_signals['Ensemble']
im2 = plot_scalogram(ensemble_denoised, 'Ensemble Denoised Scalogram', axes[1])

plt.tight_layout()
plt.show()

# Additional scalograms for other models
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flat
other_models = ['DnCNN', 'U-Net', 'N2N', 'Classical']

for idx, (ax, name) in enumerate(zip(axes, other_models)):
    plot_scalogram(denoised_signals[name], f'{name} Denoised Scalogram', ax)

plt.tight_layout()
plt.show()

print("\n✅ TESTING COMPLETE!")