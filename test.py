import numpy as np
import librosa
import matplotlib.pyplot as plt

# =========================
# Load audio
# =========================
x, sr = librosa.load("audio.wav", sr=None)

# =========================
# BASELINE: sem divisão (N = 1000)
# =========================
N_ref = 1000

X_full = np.fft.fft(x)
idx_full = np.argsort(np.abs(X_full))[-N_ref:]

X_trunc_full = np.zeros_like(X_full)
X_trunc_full[idx_full] = X_full[idx_full]

x_hat_full = np.fft.ifft(X_trunc_full).real
mse_full = np.mean((x - x_hat_full)**2)

# =========================
# Dividir em 2 partes
# =========================
n_split = 5
x_split = np.array_split(x, n_split)

# =========================
# Loop em N
# =========================
N_values = np.arange(1, int(N_ref/n_split) + 1)
mse_values = []

for N in N_values:

    x_hat_split = []

    for seg in x_split:

        # FFT
        X = np.fft.fft(seg)

        # garantir que N não exceda o tamanho do segmento
        N_eff = min(N, len(X))

        idx = np.argsort(np.abs(X))[-N_eff:]

        X_trunc = np.zeros_like(X)
        X_trunc[idx] = X[idx]

        x_hat_seg = np.fft.ifft(X_trunc).real
        x_hat_split.append(x_hat_seg)

    # reconstruir sinal completo
    x_hat = np.concatenate(x_hat_split)

    mse = np.mean((x - x_hat)**2)
    mse_values.append(mse)

# =========================
# Plot
# =========================
plt.figure()

plt.plot(N_values, mse_values, label=n_split)
plt.axhline(y=mse_full, linestyle="--", label="No split (N=1000)")

plt.xlabel("N (number of Fourier coefficients per segment)")
plt.ylabel("MSE")
plt.title("MSE vs N (Split vs Full FFT)")
plt.legend()

plt.grid()
plt.savefig("mse_vs_n.png", dpi=300)