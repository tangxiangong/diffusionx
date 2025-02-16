#![cfg_attr(feature = "nightly", feature(specialized_into_iter))]

use crate::{simulation::{Bm, ParamsBuilder}, utils::XResult};

impl std::ops::Fn for Bm {
    type Output = XResult<f64>;
    fn call(&self, args: (f64, f64)) -> Self::Output {
        let (t, x) = self.simulate(ParamsBuilder::default().time_step(args.0).duration(args.1).build().unwrap())?;
        Ok(x.last().unwrap())
    }
}
