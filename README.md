# wavelet-core

Wavelet transforms and signal denoising in pure Rust.

## Features

- **Haar wavelet** — Simplest wavelet with perfect reconstruction
- **Daubechies D4** — 4-coefficient wavelet (2 vanishing moments)
- **CWT** — Continuous wavelet transform with Morlet wavelet
- **Thresholding** — Soft and hard thresholding, universal threshold (VisuShrink)
- **Denoising** — Wavelet-based signal denoising with SNR improvement

## Modules

| Module | Description |
|--------|-------------|
| `haar` | Haar wavelet transform and inverse |
| `daubechies` | Daubechies D4 wavelet transform |
| `cwt` | Continuous wavelet transform (Morlet) |
| `threshold` | Soft/hard thresholding |
| `denoise` | Wavelet denoising utilities |

## Quick Start

```rust
use wavelet_core::{haar_transform_multilevel, haar_inverse, wavelet_denoise};

let signal = vec![1.0, 3.0, -2.0, 4.0, 0.5, -1.0, 2.0, 7.0];
let coeffs = haar_transform_multilevel(&signal);
let recovered = haar_inverse(&coeffs);

// Denoise a noisy signal
let denoised = wavelet_denoise(&noisy_signal);
```

## License

MIT OR Apache-2.0
