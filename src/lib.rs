#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

#[cfg_attr(feature = "mimalloc", global_allocator)]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub trait FloatExt:
    num_traits::Float
    + num_traits::FloatConst
    + std::fmt::Debug
    + Send
    + Sync
    + std::ops::AddAssign<Self>
    + std::ops::MulAssign<Self>
    + std::iter::Sum
{
}

impl<T> FloatExt for T where
    T: num_traits::Float
        + num_traits::FloatConst
        + std::fmt::Debug
        + Send
        + Sync
        + std::ops::AddAssign<Self>
        + std::ops::MulAssign<Self>
        + std::iter::Sum
{
}

pub trait IntExt: num_traits::PrimInt + std::fmt::Debug + Send + Sync + std::iter::Sum {}
impl<T> IntExt for T where T: num_traits::PrimInt + std::fmt::Debug + Send + Sync + std::iter::Sum {}

pub trait RealExt:
    num_traits::Num
    + num_traits::NumCast
    + std::fmt::Debug
    + Send
    + Sync
    + Copy
    + PartialOrd
    + Neg<Output = Self>
    + std::iter::Sum
{
}

impl<T> RealExt for T where
    T: num_traits::Num
        + num_traits::NumCast
        + std::fmt::Debug
        + Send
        + Sync
        + Copy
        + PartialOrd
        + Neg<Output = Self>
        + std::iter::Sum
{
}

mod error;
use std::ops::Neg;

pub use error::*;

/// Random number generation module
pub mod random;

/// Stochastic process simulation module
pub mod simulation;

/// Utility functions and algorithms
pub mod utils;

/// Visualization module
#[cfg(feature = "visualize")]
#[cfg_attr(docsrs, doc(cfg(feature = "visualize")))]
pub mod visualize;

/// GPU acceleration module
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod gpu;
