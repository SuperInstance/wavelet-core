//! Wavelet-based signal denoising.
//!
//! Uses Haar wavelet transform with soft thresholding (VisuShrink).

use crate::haar::{haar_transform_multilevel, haar_inverse};
use crate::threshold::{soft_threshold_slice, universal_threshold};

/// Denoise a signal using wavelet thresholding.
/// Uses Haar wavelet transform with soft thresholding.
pub fn wavelet_denoise(signal: &[f64]) -> Vec<f64> {
    if signal.len() < 4 {
        return signal.to_vec();
    }

    // Pad to power of 2 if needed
    let n = signal.len();
    let mut padded_len = n;
    while !padded_len.is_power_of_two() {
        padded_len += 1;
    }

    let mut padded = vec![0.0; padded_len];
    padded[..n].copy_from_slice(signal);

    // Forward transform
    let mut coeffs = haar_transform_multilevel(&padded);

    // Estimate noise and threshold detail coefficients
    let half = padded_len / 2;
    let lambda = universal_threshold(&coeffs[half..]);

    // Apply soft thresholding to detail coefficients only
    let detail_thresholded = soft_threshold_slice(&coeffs[half..], lambda);
    coeffs[half..].copy_from_slice(&detail_thresholded);

    // Inverse transform
    let denoised = haar_inverse(&coeffs);

    denoised[..n].to_vec()
}

/// Compute the signal-to-noise ratio (SNR) in dB.
pub fn compute_snr(signal: &[f64], noise: &[f64]) -> f64 {
    let signal_power: f64 = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
    let noise_power: f64 = noise.iter().map(|x| x * x).sum::<f64>() / noise.len() as f64;
    if noise_power < 1e-30 {
        return 100.0; // Essentially infinite
    }
    10.0 * (signal_power / noise_power).log10()
}

/// Compute the mean squared error between two signals.
pub fn compute_mse(signal: &[f64], reference: &[f64]) -> f64 {
    let n = signal.len().min(reference.len());
    let mse: f64 = signal[..n].iter().zip(reference[..n].iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>() / n as f64;
    mse
}

/// Advanced denoising with manual threshold control.
pub fn wavelet_denoise_with_threshold(signal: &[f64], threshold: f64) -> Vec<f64> {
    if signal.len() < 4 {
        return signal.to_vec();
    }

    let n = signal.len();
    let mut padded_len = n;
    while !padded_len.is_power_of_two() {
        padded_len += 1;
    }

    let mut padded = vec![0.0; padded_len];
    padded[..n].copy_from_slice(signal);

    let mut coeffs = haar_transform_multilevel(&padded);

    let half = padded_len / 2;
    let detail_thresholded = soft_threshold_slice(&coeffs[half..], threshold);
    coeffs[half..].copy_from_slice(&detail_thresholded);

    let denoised = haar_inverse(&coeffs);
    denoised[..n].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoise_removes_noise() {
        // Create a step signal with noise
        let mut signal = vec![0.0; 64];
        for i in 32..64 {
            signal[i] = 1.0;
        }
        // Add small noise
        let noisy: Vec<f64> = signal.iter().enumerate().map(|(i, &v)| {
            v + 0.1 * ((i as f64 * 7.3).sin() * 3.7 + (i as f64 * 2.1).cos() * 2.3)
        }).collect();

        let denoised = wavelet_denoise(&noisy);
        let mse_noisy = compute_mse(&noisy, &signal);
        let mse_denoised = compute_mse(&denoised, &signal);
        assert!(mse_denoised < mse_noisy, "Denoised MSE {} should be less than noisy MSE {}", mse_denoised, mse_noisy);
    }

    #[test]
    fn test_snr_computation() {
        let signal = vec![1.0; 100];
        let noise = vec![0.1; 100];
        let snr = compute_snr(&signal, &noise);
        assert!((snr - 20.0).abs() < 0.1, "SNR should be ~20 dB, got {}", snr);
    }

    #[test]
    fn test_mse_identical() {
        let signal = vec![1.0, 2.0, 3.0];
        let mse = compute_mse(&signal, &signal);
        assert!(mse.abs() < 1e-15, "MSE of identical signals should be 0");
    }

    #[test]
    fn test_mse_different() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 4.0];
        let mse = compute_mse(&a, &b);
        assert!((mse - 1.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_denoise_preserves_smooth() {
        // Smooth signal should be well preserved
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let denoised = wavelet_denoise(&signal);
        let mse = compute_mse(&denoised, &signal);
        assert!(mse < 0.1, "Smooth signal denoise MSE should be small: {}", mse);
    }

    #[test]
    fn test_denoise_short_signal() {
        let signal = vec![1.0, 2.0, 3.0];
        let denoised = wavelet_denoise(&signal);
        assert_eq!(denoised, signal);
    }

    #[test]
    fn test_denoise_with_manual_threshold() {
        let signal = vec![0.0; 16];
        let denoised = wavelet_denoise_with_threshold(&signal, 0.1);
        assert_eq!(denoised.len(), 16);
    }
}
