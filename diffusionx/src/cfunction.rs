//! C functions
//!
//! This module provides C functions.

#[link(name = "m")]
unsafe extern "C" {
    pub unsafe fn tgamma(x: f64) -> f64;
    pub unsafe fn tgammaf(x: f32) -> f32;
}

/// Gamma function
///
/// This function calculates the gamma function of a given number.
///
/// # Arguments
///
/// * `x` - The number to calculate the gamma function of.  
///
/// # Returns
///
/// The gamma function of the given number.
///
/// # Example
///
/// ```rust
/// use diffusionx::cfunction::gamma;
/// let x = 1.0;
/// let result = gamma(x);
/// assert_eq!(result, 1.0);
/// ```
pub fn gamma(x: f64) -> f64 {
    if x <= 0.0 {
        panic!("gamma function is not defined for non-positive numbers");
    }
    unsafe { tgamma(x) }
}

/// Gamma function for f32
///
/// This function calculates the gamma function of a given number.
///
/// # Arguments
///
/// * `x` - The number to calculate the gamma function of.
///
/// # Returns
///
/// The gamma function of the given number.
///
/// # Example
///
/// ```rust
/// use diffusionx::cfunction::gammaf;
/// let x = 1.0;
/// let result = gammaf(x);
/// assert_eq!(result, 1.0);
/// ```
pub fn gammaf(x: f32) -> f32 {
    if x <= 0.0 {
        panic!("gamma function is not defined for non-positive numbers");
    }
    unsafe { tgammaf(x) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma() {
        let x = 1.0;
        let result = gamma(x);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_gammaf() {
        let x = 1.0;
        let result = gammaf(x);
        assert_eq!(result, 1.0);
    }
}
