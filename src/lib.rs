#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Numeric bound used by simulations whose time and state values are floating point.
///
/// This trait is implemented automatically for any type that satisfies the required
/// arithmetic, constant, formatting, and thread-safety bounds. Most continuous
/// process APIs use `T: FloatExt` so they can work with `f32`, `f64`, or compatible
/// numeric types without repeating the full bound list.
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

/// Numeric bound used by discrete processes for step counters and integer states.
///
/// This trait is implemented automatically for primitive integer-like types that
/// are copyable, summable, debuggable, and safe to share across worker threads.
pub trait IntExt: num_traits::PrimInt + std::fmt::Debug + Send + Sync + std::iter::Sum {}
impl<T> IntExt for T where T: num_traits::PrimInt + std::fmt::Debug + Send + Sync + std::iter::Sum {}

/// Numeric bound used by process state values that may be integer or floating point.
///
/// `RealExt` is intentionally less restrictive than [`FloatExt`]: it supports
/// arithmetic, conversion to and from numeric types, ordering, copying, and
/// summation, but does not require floating-point-only operations such as
/// exponentials or square roots.
pub trait RealExt:
    num_traits::Num
    + num_traits::NumCast
    + std::fmt::Debug
    + Send
    + Sync
    + Copy
    + PartialOrd
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
        + std::iter::Sum
{
}

mod error;
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
