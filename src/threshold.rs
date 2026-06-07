//! Soft and hard thresholding for wavelet coefficient denoising.

/// Apply soft thresholding: T_soft(x) = sign(x) · max(|x| - λ, 0)
pub fn soft_threshold(value: f64, lambda: f64) -> f64 {
    if value > lambda {
        value - lambda
    } else if value < -lambda {
        value + lambda
    } else {
        0.0
    }
}

/// Apply hard thresholding: T_hard(x) = x if |x| > λ, else 0
pub fn hard_threshold(value: f64, lambda: f64) -> f64 {
    if value.abs() > lambda {
        value
    } else {
        0.0
    }
}

/// Apply soft thresholding to a slice of coefficients.
pub fn soft_threshold_slice(coeffs: &[f64], lambda: f64) -> Vec<f64> {
    coeffs.iter().map(|&x| soft_threshold(x, lambda)).collect()
}

/// Apply hard thresholding to a slice of coefficients.
pub fn hard_threshold_slice(coeffs: &[f64], lambda: f64) -> Vec<f64> {
    coeffs.iter().map(|&x| hard_threshold(x, lambda)).collect()
}

/// Estimate the noise threshold using the universal threshold (VisuShrink).
/// λ = σ̂ · √(2 · ln(N))
pub fn universal_threshold(coeffs: &[f64]) -> f64 {
    let n = coeffs.len() as f64;
    let sigma = estimate_noise_std(coeffs);
    sigma * (2.0 * n.ln()).sqrt()
}

/// Estimate noise standard deviation using median absolute deviation (MAD).
/// σ̂ = median(|coeffs|) / 0.6745
pub fn estimate_noise_std(coeffs: &[f64]) -> f64 {
    let mut abs_vals: Vec<f64> = coeffs.iter().map(|x| x.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if abs_vals.is_empty() {
        0.0
    } else {
        abs_vals[abs_vals.len() / 2]
    };
    median / 0.6745
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_threshold_zero_lambda() {
        assert!((soft_threshold(5.0, 0.0) - 5.0).abs() < 1e-14);
        assert!((soft_threshold(-3.0, 0.0) - (-3.0)).abs() < 1e-14);
    }

    #[test]
    fn test_soft_threshold_shrinkage() {
        let result = soft_threshold(5.0, 2.0);
        assert!((result - 3.0).abs() < 1e-14);
        let result_neg = soft_threshold(-5.0, 2.0);
        assert!((result_neg - (-3.0)).abs() < 1e-14);
    }

    #[test]
    fn test_soft_threshold_zero() {
        assert!((soft_threshold(1.0, 2.0)).abs() < 1e-14);
        assert!((soft_threshold(-1.0, 2.0)).abs() < 1e-14);
    }

    #[test]
    fn test_hard_threshold_keeps() {
        assert!((hard_threshold(5.0, 2.0) - 5.0).abs() < 1e-14);
        assert!((hard_threshold(-5.0, 2.0) - (-5.0)).abs() < 1e-14);
    }

    #[test]
    fn test_hard_threshold_zeros() {
        assert!((hard_threshold(1.0, 2.0)).abs() < 1e-14);
        assert!((hard_threshold(-1.0, 2.0)).abs() < 1e-14);
    }

    #[test]
    fn test_soft_threshold_slice() {
        let coeffs = [1.0, 5.0, -3.0, -0.5, 2.0];
        let result = soft_threshold_slice(&coeffs, 1.0);
        assert!((result[1] - 4.0).abs() < 1e-14);
        assert!(result[3].abs() < 1e-14);
    }

    #[test]
    fn test_hard_threshold_slice() {
        let coeffs = [1.0, 5.0, -3.0, -0.5, 2.0];
        let result = hard_threshold_slice(&coeffs, 1.5);
        assert!(result[0].abs() < 1e-14);
        assert!((result[1] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_universal_threshold_positive() {
        let coeffs = [0.1, -0.2, 0.15, -0.05, 0.3, -0.1, 0.2, -0.25];
        let lambda = universal_threshold(&coeffs);
        assert!(lambda > 0.0, "Universal threshold should be positive");
    }

    #[test]
    fn test_estimate_noise_std() {
        // Known noise level
        let noise = [0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.05, -0.05];
        let sigma = estimate_noise_std(&noise);
        assert!(sigma > 0.0, "Noise std should be positive");
    }
}
