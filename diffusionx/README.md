# DiffusionX

> [!NOTE]
> Development is in progress. DiffusionX is a multi-threaded high-performance Rust library for random number/stochastic process simulation.

## Usage
```rust
use diffusionx::simulation::{Bm, Simulation, Functional};

// Brownian motion simulation
let bm = Bm::new(0.0, 1.0, 1.0)?;  // Create Brownian motion object: initial position 0, diffusion coefficient 1, duration 1
let time_step = 0.01;  // Time step
let (times, positions) = bm.simulate(time_step)?;  // Simulate Brownian motion trajectory

// Monte Carlo simulation of Brownian motion statistics
let mean = bm.mean(time_step, 1000)?;  // Mean  bm.raw_moment(time_step, 1, 1000)?;
let msd = bm.msd(time_step, 1000)?;  // Mean square displacement  bm.central_moment(time_step, 2, 1000)?;

// First passage time of Brownian motion
let max_duration = 1000; // if over this duration, the simulation will be terminated and return None
let fpt = bm.fpt(time_step, (-1.0, 1.0), max_duration)?; 
// or
let fpt = FirstPassageTime::new(&bm, (-1.0, 1.0))?;
let fpt_result = fpt.simulate(max_duration, time_step)?;
```
## Extensibility

DiffusionX is designed with a trait-based system for high extensibility:

### Core Traits

- `Stochastic`: Base trait for stochastic processes
- `Simulation`: Core trait for process simulation, defining simulation methods
- `Moment`: Trait for statistical moments calculation, including raw and central moments
- `Trajectory`: Trait for trajectory handling, providing trajectory-related functionalities

### Feature Extension

1. Adding New Stochastic Processes:
   ```rust
   #[derive(Clone)]
   struct MyProcess {
       // Your parameters
   }
   
   impl Stochastic for MyProcess {}
   
   impl Simulation for MyProcess {
       fn simulate(&self, duration: impl Into<f64>, time_step: f64) -> XResult<(Vec<f64>, Vec<f64>)> {
           // Implement your simulation logic
           t, x
       }
   }
   ```

2. Automatic Feature Acquisition:
   - Get `Trajectory` and `Moment` traits functionality automatically by implementing `Simulation` trait
   - Direct access to functional structures like `FirstPassageTime`
   - Built-in support for moment statistics calculation
Example
```rust
let myprocess = MyProcess::default();
let traj = myprocess.duration(10)?;
let mean = traj.raw_moment(1, 1000, time_step)?;
let msd = traj.central_moment(2, 1000, time_step)?;
let fpt = FirstPassageTime::new(&myprocess, (-1.0, 1.0))?;
let fpt_result = fpt.simulate(max_duration, time_step)?;
let fpt_mean = fpt.raw_moment(1, 1000, time_step)?;
```
3. Parallel Computing Support:
   - Automatic parallel computation support for all types implementing `Simulation`
   - Default parallel strategy for statistical calculations


## Progress
### Random Number Generation

- [x] Normal distribution
- [x] Uniform distribution
- [x] Exponential distribution
- [x] Poisson distribution
- [x] Alpha-stable distribution

### Stochastic Processes

- [x] Brownian motion
- [ ] Alpha-stable Lévy process
- [ ] Fractional Brownian motion
- [ ] Poisson process
- [ ] Compound Poisson process
- [ ] Langevin equation

## Benchmark

### Test Results

Generating random array of length `10_000_000`

|               | Standard Normal | Uniform [0, 1] |   Stable   |
| :-----------: | :-------------: | :------------: | :--------: |
|  DiffusionX   |    23.811 ms    |   20.450 ms    | 273.68 ms  |
|     Julia     |    28.748 ms    |    9.748 ms    | 713.955 ms |
| NumPy / SciPy |     295 ms      |    81.2 ms     |   3.39 s   |
|     Numba     |        -        |       -        |   1.52 s   |

### Test Environment

#### Hardware Configuration
- Device Model: MacBook Pro 13-inch (2020)
- Processor: Intel Core i5-1038NG7 @ 2.0GHz (4 cores 8 threads)
- Memory: 16GB LPDDR4X 3733MHz

#### Software Environment
- Operating System: macOS Sequoia 15.3
- Rust: 1.85.0-beta.7
- Python: 3.12
- Julia: 1.11
- NumPy: 2.2.2
- SciPy: 1.15.1

## License

This project is dual-licensed under:

* [MIT License](https://opensource.org/licenses/MIT)
* [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

You can choose to use either license. 