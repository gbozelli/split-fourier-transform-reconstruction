import numpy as np
import librosa
import soundfile as sf

# load audio
x, sr = librosa.load("audio.wav", sr=None)

# FFT
X = np.fft.fft(x)

# keep N largest coefficients
N = 100000
idx = np.argsort(np.abs(X))[-N:]
X_trunc = np.zeros_like(X)
X_trunc[idx] = X[idx]

# reconstruct signal
x_hat = np.fft.ifft(X_trunc).real

mse = np.mean((x - x_hat)**2)
sf.write("reconstructed.wav", x_hat, sr)

print('MSE is', mse)
print('The wave was reconstructed with', N, 'biggest terms')

import numpy as np

m = np.arange(0,1000,1)
N = np.arange(0,1000,1)

x_split = np.array_split(x, m)

x_hat_split = []

for seg in x_split:

    # FFT
    X = np.fft.fft(seg)

    # indices of largest coefficients
    idx = np.argsort(np.abs(X))[-N:]

    # truncated spectrum (keep positions!)
    X_trunc = np.zeros_like(X)
    X_trunc[idx] = X[idx]

    # reconstruct
    x_hat_seg = np.fft.ifft(X_trunc).real

    x_hat_split.append(x_hat_seg)

# merge segments
x_hat = np.concatenate(x_hat_split)

mse = np.mean((x - x_hat)**2)
sf.write("reconstructed_sftr.wav", x_hat, sr)

print('MSE is', mse)
print('The wave was reconstructed with', N, 'biggest terms and split method with', m,'splits')