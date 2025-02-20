//! This module provides the callable trait for the simulation.
//!
//! **Need `nightly` rustc to compile.**
//!
//! # Features
//!
//! - `callable`: Enable the callable trait for the simulation.
//!
//! # Examples
//!
//! ```rust
//! use diffusionx::simulation::prelude::*;
//! use diffusionx::simulation::callable::*;
//! ```
//!
//! ```rust
//! let sp = BrownianMotion::new(1.0);
//! let traj = sp.callable(1.0);
//! ```

#![feature(unboxed_closures)]
#![feature(fn_traits)]

use crate::{XResult, simulation::prelude::*};

impl<SP: ContinuousProcess, D: Into<f64>> FnOnce<(D,)> for SP {
    type Output = XResult<ContinuousTrajectory<SP>>;
    extern "rust-call" fn call_once(self, args: (D,)) -> Self::Output {
        ContinuousTrajectory::new(self, args.0)
    }
}

impl<SP: ContinuousProcess, D: Into<f64>> FnMut<(D,)> for SP {
    extern "rust-call" fn call_mut(&mut self, args: (D,)) -> Self::Output {
        ContinuousTrajectory::new(self, args.0)
    }
}

impl<SP: ContinuousProcess, D: Into<f64>> Fn<(D,)> for ContinuousTrajectory<SP> {
    extern "rust-call" fn call(&self, args: (D,)) -> Self::Output {
        ContinuousTrajectory::new(self, args.0)
    }
}

