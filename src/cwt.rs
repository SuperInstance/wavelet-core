//! Continuous Wavelet Transform (CWT) basics.
//!
//! Provides a basic CWT implementation using the Morlet wavelet.

use std::f64::consts::{PI, SQRT_2};

/// Morlet wavelet (complex valued) at a given time position.
/// ψ(t) = π^{-1/4} · e^{iω₀t} · e^{-t²/2}
pub fn morlet(t: f64, omega0: f64) -> (f64, f64) {
    let norm = PI.powf(-0.25);
    let gaussian = (-t * t / 2.0).exp();
    let phase = omega0 * t;
    let re = norm * gaussian * phase.cos();
    let im = norm * gaussian * phase.sin();
    (re, im)
}

/// Compute the CWT of a signal at a given scale using the Morlet wavelet.
/// Returns (real_part, imaginary_part) of the transform.
pub fn cwt_at_scale(signal: &[f64], scale: f64, omega0: f64) -> Vec<(f64, f64)> {
    let dt = 1.0;
    let n = signal.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let t_center = i as f64;
        let mut re_sum = 0.0;
        let mut im_sum = 0.0;

        // Convolve with scaled wavelet
        let support = (5.0 * scale) as usize;
        let start = (t_center as usize).saturating_sub(support);
        let end = (t_center as usize + support + 1).min(n);

        for (j, &sig_val) in signal.iter().enumerate().take(end).skip(start) {
            let t = (j as f64 - t_center) * dt / scale;
            let (wr, wi) = morlet(t, omega0);
            re_sum += sig_val * wr;
            im_sum += sig_val * wi;
        }

        let norm = 1.0 / SQRT_2 * scale.sqrt();
        result.push((re_sum * norm * dt / scale, im_sum * norm * dt / scale));
    }

    result
}

/// Compute the CWT magnitude at a given scale.
pub fn cwt_magnitude_at_scale(signal: &[f64], scale: f64, omega0: f64) -> Vec<f64> {
    cwt_at_scale(signal, scale, omega0)
        .iter()
        .map(|(re, im)| re.hypot(*im))
        .collect()
}

/// Compute the CWT scalogram over multiple scales.
/// Returns a vector of (scale, magnitudes).
pub fn scalogram(signal: &[f64], scales: &[f64], omega0: f64) -> Vec<(f64, Vec<f64>)> {
    scales.iter().map(|&scale| {
        let mag = cwt_magnitude_at_scale(signal, scale, omega0);
        (scale, mag)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morlet_at_zero() {
        let (re, im) = morlet(0.0, 6.0);
        assert!((re - PI.powf(-0.25)).abs() < 1e-14);
        assert!(im.abs() < 1e-14);
    }

    #[test]
    fn test_morlet_decay() {
        let (re_0, _) = morlet(0.0, 6.0);
        let (re_5, _) = morlet(5.0, 6.0);
        assert!(re_5.abs() < re_0.abs() * 0.01);
    }

    #[test]
    fn test_morlet_symmetric_real() {
        let (re_pos, _) = morlet(1.0, 6.0);
        let (re_neg, _) = morlet(-1.0, 6.0);
        // Real part should be symmetric for omega0=0, but asymmetric for nonzero
        // At least magnitudes should be similar
        assert!((re_pos - re_neg).abs() < 0.01 * re_pos.abs().max(re_neg.abs()) + 1e-15);
    }

    #[test]
    fn test_cwt_output_length() {
        let signal = vec![1.0; 32];
        let result = cwt_at_scale(&signal, 2.0, 6.0);
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_cwt_magnitude_nonnegative() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let mag = cwt_magnitude_at_scale(&signal, 1.0, 6.0);
        for &m in &mag {
            assert!(m >= 0.0, "CWT magnitude should be non-negative");
        }
    }

    #[test]
    fn test_scalogram_scales() {
        let signal = vec![1.0; 16];
        let scales = vec![1.0, 2.0, 4.0, 8.0];
        let sg = scalogram(&signal, &scales, 6.0);
        assert_eq!(sg.len(), 4);
        for (scale, mags) in &sg {
            assert_eq!(mags.len(), 16);
            assert!(*scale > 0.0);
        }
    }

    #[test]
    fn test_cwt_constant_signal() {
        // CWT of a constant signal should produce non-trivial output at large scales
        let signal = vec![5.0; 64];
        let mag_high = cwt_magnitude_at_scale(&signal, 10.0, 6.0);
        let energy_high: f64 = mag_high.iter().map(|x| x * x).sum::<f64>();
        assert!(energy_high > 0.0, "CWT of constant signal should have nonzero energy");
    }
}
