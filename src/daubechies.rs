//! Daubechies D4 wavelet transform and inverse.
//!
//! Uses the 4-coefficient Daubechies wavelet (2 vanishing moments).

/// Daubechies D4 low-pass decomposition filter coefficients (Db2 scaling).
pub const D4_DECOMP_LO: [f64; 4] = [
    0.4829629131445341,
    0.8365163037378079,
    0.2241438680420134,
    -0.1294095225512603,
];

/// Daubechies D4 high-pass decomposition filter coefficients (Db2 wavelet).
/// g[k] = (-1)^k * h[L-1-k]
pub const D4_DECOMP_HI: [f64; 4] = [
    -0.1294095225512603,
    -0.2241438680420134,
    0.8365163037378079,
    -0.4829629131445341,
];

/// Daubechies D4 low-pass reconstruction filter coefficients.
pub const D4_RECON_LO: [f64; 4] = [
    -0.1294095225512603,
    0.2241438680420134,
    0.8365163037378079,
    0.4829629131445341,
];

/// Daubechies D4 high-pass reconstruction filter coefficients.
pub const D4_RECON_HI: [f64; 4] = [
    -0.4829629131445341,
    0.8365163037378079,
    -0.2241438680420134,
    -0.1294095225512603,
];

/// Normalize a signal to have unit energy.
pub fn normalize(signal: &[f64]) -> Vec<f64> {
    let energy: f64 = signal.iter().map(|x| x * x).sum();
    if energy < 1e-30 {
        return signal.to_vec();
    }
    let scale = 1.0 / energy.sqrt();
    signal.iter().map(|x| x * scale).collect()
}

/// Forward Daubechies D4 wavelet transform (single level).
/// Signal length must be a power of 2 and >= 4.
pub fn daubechies4_transform(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(n >= 4, "Signal length must be >= 4");
    assert!(n.is_power_of_two(), "Signal length must be a power of 2");

    let mut coeffs = vec![0.0; n];
    let half = n / 2;

    for i in 0..half {
        let i2 = 2 * i;
        let mut approx = 0.0;
        let mut detail = 0.0;
        for k in 0..4 {
            let idx = (i2 + k) % n;
            approx += D4_DECOMP_LO[k] * signal[idx];
            detail += D4_DECOMP_HI[k] * signal[idx];
        }
        coeffs[i] = approx;
        coeffs[half + i] = detail;
    }

    coeffs
}

/// Multi-level Daubechies D4 wavelet transform.
pub fn daubechies4_multilevel(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(n >= 4 && n.is_power_of_two());

    let mut data = signal.to_vec();
    let mut coeffs = vec![0.0; n];
    let mut len = n;

    while len >= 4 {
        let half = len / 2;
        let mut approx = vec![0.0; half];
        let mut detail = vec![0.0; half];

        for i in 0..half {
            let i2 = 2 * i;
            for k in 0..4 {
                let idx = (i2 + k) % len;
                approx[i] += D4_DECOMP_LO[k] * data[idx];
                detail[i] += D4_DECOMP_HI[k] * data[idx];
            }
        }

        coeffs[half..len].copy_from_slice(&detail);
        data[..half].copy_from_slice(&approx);
        len = half;
    }
    coeffs[..len].copy_from_slice(&data[..len]);

    coeffs
}

/// Inverse Daubechies D4 wavelet transform (single level).
pub fn daubechies4_inverse(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    assert!(n >= 4 && n.is_power_of_two());

    let half = n / 2;
    let approx = &coeffs[..half];
    let detail = &coeffs[half..];

    // Reconstruction: x[n] = W^T @ coeffs
    // For each output n, sum contributions from all analysis rows
    let mut signal = vec![0.0; n];
    for j in 0..half {
        for k in 0..4 {
            let out_idx = (2 * j + k) % n;
            signal[out_idx] += D4_DECOMP_LO[k] * approx[j];
            signal[out_idx] += D4_DECOMP_HI[k] * detail[j];
        }
    }

    signal
}

/// Inverse multi-level Daubechies D4 transform.
pub fn daubechies4_inverse_multilevel(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    assert!(n >= 4 && n.is_power_of_two());

    let mut data = coeffs.to_vec();
    let mut len = 4; // start from smallest level

    while len <= n {
        let half = len / 2;
        let mut new_data = vec![0.0; len];

        for j in 0..half {
            for k in 0..4 {
                let out_idx = (2 * j + k) % len;
                new_data[out_idx] += D4_DECOMP_LO[k] * data[j];
                new_data[out_idx] += D4_DECOMP_HI[k] * data[half + j];
            }
        }

        data[..len].copy_from_slice(&new_data);
        len *= 2;
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d4_perfect_reconstruction_single_level() {
        let signal = vec![1.0, 3.0, -2.0, 4.0, 0.5, -1.0, 2.0, 7.0];
        let coeffs = daubechies4_transform(&signal);
        let inv = daubechies4_inverse(&coeffs);
        for i in 0..signal.len() {
            assert!((inv[i] - signal[i]).abs() < 1e-10, "Mismatch at {}: {} vs {}", i, inv[i], signal[i]);
        }
    }

    #[test]
    fn test_d4_filter_sum() {
        // Low-pass filter should sum to sqrt(2)
        let sum: f64 = D4_DECOMP_LO.iter().sum();
        assert!((sum - std::f64::consts::SQRT_2).abs() < 0.01, "Low-pass sum: {}", sum);
    }

    #[test]
    fn test_d4_highpass_sum_zero() {
        let sum: f64 = D4_DECOMP_HI.iter().sum();
        assert!(sum.abs() < 0.01, "High-pass sum should be ~0: {}", sum);
    }

    #[test]
    fn test_d4_orthogonality() {
        // Inner product of lowpass and highpass should be ~0
        let dot: f64 = D4_DECOMP_LO.iter().zip(D4_DECOMP_HI.iter()).map(|(&a, &b)| a * b).sum();
        assert!(dot.abs() < 0.01, "Filters should be orthogonal: dot={}", dot);
    }

    #[test]
    fn test_d4_vanishing_moments() {
        // D4 has 2 vanishing moments: should annihilate polynomials of degree < 2
        let linear = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = daubechies4_transform(&linear);
        let half = linear.len() / 2;
        // Detail coefficients should be small for smooth signals
        let detail_energy: f64 = coeffs[half..].iter().map(|x| x * x).sum::<f64>();
        let total_energy: f64 = coeffs.iter().map(|x| x * x).sum();
        let ratio = detail_energy / total_energy;
        assert!(ratio < 0.5, "Detail energy ratio too high: {}", ratio);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        let energy: f64 = n.iter().map(|x| x * x).sum();
        assert!((energy - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_d4_multilevel_length() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = daubechies4_multilevel(&signal);
        assert_eq!(coeffs.len(), 8);
    }
}
