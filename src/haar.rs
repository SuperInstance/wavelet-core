//! Haar wavelet transform and inverse.
//!
//! The Haar wavelet is the simplest wavelet, using pairs of adjacent samples.

/// Forward Haar wavelet transform.
/// Input length must be a power of 2.
/// Returns (approximation coefficients, detail coefficients) interleaved:
/// [a0, d0, a1, d1, ...] at each level.
pub fn haar_transform(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(n.is_power_of_two(), "Signal length must be a power of 2");

    let mut data = signal.to_vec();
    let mut result = vec![0.0; n];
    let mut len = n;

    while len > 1 {
        let half = len / 2;
        for i in 0..half {
            let a = (data[2 * i] + data[2 * i + 1]) / std::f64::consts::SQRT_2;
            let d = (data[2 * i] - data[2 * i + 1]) / std::f64::consts::SQRT_2;
            result[i] = a;
            result[half + i] = d;
        }
        data[..len].copy_from_slice(&result[..len]);
        len = half;
    }

    // Copy back properly: result is already in the right format
    result
}

/// Multi-level Haar wavelet transform returning coefficients at all levels.
pub fn haar_transform_multilevel(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(n.is_power_of_two(), "Signal length must be a power of 2");

    let mut data = signal.to_vec();
    let mut coeffs = signal.to_vec();
    let mut len = n;

    while len > 1 {
        let half = len / 2;
        let mut approx = vec![0.0; half];
        let mut detail = vec![0.0; half];

        for i in 0..half {
            approx[i] = (data[2 * i] + data[2 * i + 1]) / std::f64::consts::SQRT_2;
            detail[i] = (data[2 * i] - data[2 * i + 1]) / std::f64::consts::SQRT_2;
        }

        coeffs[half..len].copy_from_slice(&detail);
        data[..half].copy_from_slice(&approx);
        len = half;
    }
    coeffs[0] = data[0];

    coeffs
}

/// Inverse Haar wavelet transform from multi-level coefficients.
pub fn haar_inverse(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    assert!(n.is_power_of_two(), "Coefficient length must be a power of 2");

    let mut data = coeffs.to_vec();
    let mut len = 1;

    while len < n {
        let double = len * 2;
        let mut new_data = vec![0.0; double];

        for i in 0..len {
            let a = data[i];
            let d = data[len + i];
            new_data[2 * i] = (a + d) / std::f64::consts::SQRT_2;
            new_data[2 * i + 1] = (a - d) / std::f64::consts::SQRT_2;
        }

        data[..double].copy_from_slice(&new_data);
        len = double;
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_constant_signal() {
        let signal = vec![4.0; 8];
        let coeffs = haar_transform_multilevel(&signal);
        // All detail coefficients should be zero
        for i in 1..8 {
            assert!(coeffs[i].abs() < 1e-14, "Detail coeff {} should be 0: {}", i, coeffs[i]);
        }
    }

    #[test]
    fn test_haar_perfect_reconstruction() {
        let signal = vec![1.0, 3.0, -2.0, 4.0, 0.5, -1.0, 2.0, 7.0];
        let coeffs = haar_transform_multilevel(&signal);
        let reconstructed = haar_inverse(&coeffs);
        for i in 0..signal.len() {
            assert!((reconstructed[i] - signal[i]).abs() < 1e-12, "Mismatch at {}", i);
        }
    }

    #[test]
    fn test_haar_energy_conservation() {
        let signal = vec![1.0, 3.0, -2.0, 4.0, 0.5, -1.0, 2.0, 7.0];
        let coeffs = haar_transform_multilevel(&signal);
        let energy_signal: f64 = signal.iter().map(|x| x * x).sum();
        let energy_coeffs: f64 = coeffs.iter().map(|x| x * x).sum();
        assert!((energy_signal - energy_coeffs).abs() < 1e-10,
            "Energy not conserved: {} vs {}", energy_signal, energy_coeffs);
    }

    #[test]
    fn test_haar_step_function() {
        let signal = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let coeffs = haar_transform_multilevel(&signal);
        // Detail coefficients should be nonzero at some level
        let detail_energy: f64 = coeffs[1..].iter().map(|x| x * x).sum();
        assert!(detail_energy > 0.1, "Should detect step, detail_energy={}", detail_energy);
    }

    #[test]
    fn test_haar_length_2() {
        let signal = vec![3.0, 1.0];
        let coeffs = haar_transform_multilevel(&signal);
        let inv = haar_inverse(&coeffs);
        assert!((inv[0] - 3.0).abs() < 1e-14);
        assert!((inv[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    #[should_panic(expected = "power of 2")]
    fn test_haar_non_power_of_2_panics() {
        haar_transform_multilevel(&[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_haar_inverse_matches_forward() {
        let signal = vec![2.0, -1.0, 3.5, 0.0, -4.0, 1.0, 2.0, -3.0, 5.0, -2.0, 1.5, 0.5, -1.0, 3.0, 2.5, -0.5];
        let coeffs = haar_transform_multilevel(&signal);
        let recovered = haar_inverse(&coeffs);
        for i in 0..signal.len() {
            assert!((recovered[i] - signal[i]).abs() < 1e-12);
        }
    }
}
