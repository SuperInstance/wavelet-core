//! # Wavelet Core
//!
//! Wavelet transforms including Haar wavelet, Daubechies D4, continuous wavelet
//! transform basics, soft/hard thresholding, and signal denoising.

pub mod haar;
pub mod daubechies;
pub mod cwt;
pub mod threshold;
pub mod denoise;

pub use haar::{haar_transform, haar_inverse};
pub use daubechies::{daubechies4_transform, daubechies4_inverse};
pub use threshold::{soft_threshold, hard_threshold};
pub use denoise::wavelet_denoise;

/// Pad a signal to the next power of 2 length.
pub fn pad_to_power_of_2(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mut padded = n;
    while !padded.is_power_of_two() {
        padded += 1;
    }
    let mut result = vec![0.0; padded];
    result[..n].copy_from_slice(signal);
    result
}
