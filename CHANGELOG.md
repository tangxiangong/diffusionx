# Changelog

All notable changes to this project will be documented in this file.

## [0.3.7] - 2025-05-06

### 🚀 Features

- Add ensure_output_dir function to handle output directory creation
- Integrate ensure_output_dir in write_csv function

### 🚜 Refactor

- Remove ensure_output_dir function from draw.rs

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG for version 0.3.6
- Bump version to 0.3.7 in Cargo.toml

## [0.3.6] - 2025-05-06

### 🚀 Features

- Add Geometric Brownian motion simulation
- Add Geometric Brownian motion module
- Enhance ContinuousProcess and DiscreteProcess traits
- Add CSV error handling and CSV writing functionality
- Add CSV feature to default dependencies in Cargo.toml

### 🐛 Bug Fixes

- Mark GeometricBrownianMotion as completed in TODO list

### 🚜 Refactor

- Remove unused methods from continuous process simulations
- Remove unused simulation methods from RandomWalk
- Remove unused simulation methods from BirthDeath, CTRW, and Poisson

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG for version 0.3.5
- Bump version to 0.3.6 in Cargo.toml

## [0.3.5] - 2025-04-29

### 🚀 Features

- Add Brownian meander simulation implementation
- Add Brownian meander module to continuous simulation
- Add Asymmetric Lévy process simulation
- Add Cauchy and Asymmetric Cauchy process simulations
- Add Gamma distribution random number generation
- Add Inverse process for continuous processes
- Implement Gamma process simulation

### 🐛 Bug Fixes

- Improve error handling and documentation in random distributions
- Update README documentation path for consistency

### 🚜 Refactor

- Simplify Brownian excursion simulation logic
- Improve code readability and structure in Brownian meander simulation

### 📚 Documentation

- Update README for random number generation and visualization
- Update README to include Brownian meander
- Update documentation to include Brownian meander in simulation module
- Mark Brownian meander as completed in TODO list
- Update README for visualization configuration consistency
- Add Cauchy process to README
- Update simulation module documentation to include Cauchy process
- Mark Cauchy process as completed in TODO list
- Add Chinese version link to README
- Update README for improved clarity and links
- Update random module documentation to include Gamma distribution
- Update Levy and Subordinator simulation examples for clarity
- Add Gamma process to README
- Add Gamma process to simulation documentation
- Mark Gamma process as completed in TODO list

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG for version 0.3.4
- Update authors and description in Cargo.toml
- Bump version to 0.3.5 in Cargo.toml

## [0.3.4] - 2025-04-24

### 🚀 Features

- Add FFT planner lock error to XError enum
- Enhance CirculantEmbedding with eigenvalue caching and FFT plans
- Enhance CirculantEmbedding with eigenvalue computation and caching
- Add Brownian excursion example simulation
- Add Brownian excursion simulation implementation
- Add visualization test for Brownian motion
- Update trajectory visualization to SVG format

### 🐛 Bug Fixes

- Replace unwrap with ? in simulation methods for better error handling
- Correct occupation time parameter in Brownian excursion test

### 💼 Other

- Add Brownian bridge example simulation

### 🚜 Refactor

- Update simulation return types from PointPair to Pair
- Simplify type casting in simulation methods
- Update CTRW simulation methods for improved clarity
- Update PlotConfig defaults and improve data handling

### 📚 Documentation

- Update documentation for random and simulation modules
- Update TODO list to mark BrownianBridge as completed
- Add Brownian excursion to the README files
- Add Brownian excursion description to simulation module
- Update mod.rs to include Brownian excursion in module exports
- Update TODO list to mark BrownianExcursion as completed

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG for version 0.3.3
- Comment out unused pre-commit hooks for Rust
- Update dependency versions in Cargo.toml
- Update GitHub Actions workflow to include stable branch
- Update dependencies and version in Cargo.toml

## [0.3.3] - 2025-04-15

### 🚀 Features

- Add deny.toml configuration file for cargo-deny
- Add pre-commit configuration for Rust and Python
- Implement Brownian bridge simulation

### 📚 Documentation

- Update mod.rs to include Brownian bridge in the simulation module documentation
- Update README files to include Brownian bridge in the list of processes

### ⚙️ Miscellaneous Tasks

- Bump version to 0.3.2 in Cargo.toml
- Update CHANGELOG for version 0.3.2
- Update GitHub Actions workflow to include specific file paths for Rust files
- Comment out unused pre-commit hooks for Rust
- Bump version to 0.3.3 in Cargo.toml

## [0.3.2] - 2025-04-05

### 🚀 Features

- Enhance Fbm simulation and CirculantEmbedding functionality
- Add error handling for non-positive definite matrices

### 🚜 Refactor

- Update default implementations for simulation structs
- Clean up error messages in error.rs
- Update error messages for clarity
- Clean up error messages in error.rs

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG for version 0.3.1
- Update GitHub Actions workflow to install dependencies
- Add optional dependency once_cell in Cargo.toml

## [0.3.1] - 2025-03-31

### 🚀 Features

- Add TODO list for stochastic processes
- Introduce discrete process and trajectory traits
- Add discrete module for simulation
- Add random walk module to discrete simulation
- Implement LatticeRandomWalk for discrete simulation
- Implement RandomWalk struct for discrete simulation

### 🐛 Bug Fixes

- Correct capitalization in Chinese and English README files

### 🚜 Refactor

- Change module visibility to public in simulation files
- Update types in discrete simulation traits

### 📚 Documentation

- Enhance error handling documentation for diffusionx crate
- Enhance documentation and structure for stochastic processes
- Update benchmark section in README files
- Remove software environment details from Chinese README
- Improve comments in Brownian motion simulation
- Update README files to include DiscreteProcess trait
- Update README files to include Random Walk

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG for version 0.3.0
- Remove unused benchmark for random number generation
- Bump version to 0.3.1 in Cargo.toml

## [0.3.0] - 2025-03-22

### 🚀 Features

- Update CHANGELOG for version 0.2.2
- Add plotting functions for continuous and point trajectories
- Refactor simulation modules and add continuous processes
- Update documentation for version 0.2.2 and add visualization examples

### 🐛 Bug Fixes

- Update documentation for functional distribution simulation

### ⚙️ Miscellaneous Tasks

- Bump version to 0.3.0 in Cargo.toml

## [0.2.2] - 2025-03-20

### 🚀 Features

- Implement Exponential, Normal, and Poisson distributions with error handling
- Add InvalidParameters error variant to XError enum
- Enhance visualization of Ornstein-Uhlenbeck process
- Enhance PlotConfig and visualization functionality
- Update version and add visualization feature

### 💼 Other

- Add examples for various stochastic processes

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG for version 0.2.1
- Update GitHub Actions workflow to limit branches for push and pull requests
- Update .gitignore to include tmp directory

## [0.2.1] - 2025-03-18

### 🚀 Features

- Add visualization feature and enhance error handling

### ⚙️ Miscellaneous Tasks

- Bump version to 0.2.1 and update dependencies

## [0.2.0] - 2025-03-18

### 🚀 Features

- *(visualization)* Add optional visualization feature with Plotters integration
- *(visualization)* Enhance visualization capabilities with new PlotConfig and plotter modules
- *(error)* Add VisualizationError and PlotterError enums for enhanced error handling in visualization
- *(visualization)* Add time_step field and drawing trait for visualization
- *(error)* Add InvalidColor variant to PlotterError enum for enhanced error reporting
- *(visualization)* Enhance PlotConfig with new fields and color handling
- *(visualization)* Enhance PlotConfig and Visualize trait for improved plotting
- Update version to 0.2.0 and add visualization feature

### 🐛 Bug Fixes

- *(visualization)* Update PlotConfig struct for improved type handling
- *(tests)* Update occupation time assertions for stability

### 🚜 Refactor

- *(functions)* Remove sine and cosine functions for pi calculations
- *(functions)* Remove gamma function documentation
- *(visualization)* Update .gitignore and remove plotter module
- *(visualization)* Simplify RGBColor conversion in Color enum
- *(visualization)* Remove visualization module and related dependencies

### 📚 Documentation

- *(changelog)* Update CHANGELOG.md for version 0.1.9

### ⚙️ Miscellaneous Tasks

- *(dependencies)* Simplify Cargo.toml by removing features section and updating optional dependencies to required
- *(visualization)* Comment out plotter module and its usage
- *(workflow)* Update GitHub Actions to include 'plot' branch for CI

## [0.1.9] - 2025-03-12

### 🚀 Features

- *(simulation)* Add Birth-death process simulation module

### 📚 Documentation

- *(readme)* Update roadmap with Birth-death process feature

### ⚙️ Miscellaneous Tasks

- *(changelog)* Update CHANGELOG.md for version 0.1.8
- *(version)* Bump library version to 0.1.9 in Cargo.toml

## [0.1.8] - 2025-03-08

### 🚀 Features

- *(simulation)* Add Ornstein-Uhlenbeck process simulation module

### 💼 Other

- *(rust-ver 0.1.7)* Bump version to 0.1.7 and update changelog

### 🚜 Refactor

- Restructure project and remove Python bindings to a new repo
- *(simulation)* Generalize Langevin and Generalized Langevin structs with generic function types

### 📚 Documentation

- Refactor README for improved clarity and content
- Update README language links and formatting
- *(readme)* Enhance README with comprehensive library overview and refined examples

### 🧪 Testing

- Refactor test suites for simulation modules
- Modify Langevin test case to remove strict assertion

## [Rust-v0.1.7] - 2025-03-05

### 🚀 Features

- *(py-diffusionx)* Add support for FBM, CTRW and Langevin processes
- *(levy_walk)* Add Levy walk simulation module

### 🚜 Refactor

- Format

### 📚 Documentation

- *(changelog)* Add entries for Python-v0.1.3 and Rust-v0.1.6 releases

## [Rust-v0.1.6] - 2025-03-04

### 🚀 Features

- *(CTRW)* Add continuous-time random walk model

### 💼 Other

- Bump Rust crate version 0.1.6

### 📚 Documentation

- *(CHANGELOG)* Update changelog for Rust v0.1.5 release
- *(bm,fbm)* Improve documentation and code formatting

## [Rust-v0.1.5] - 2025-03-01

### 🚀 Features

- *(Rust)* Add Circulant Embedding Method for Gaussian Random Fields
- *(Simulation)* Implement Fractional Brownian Motion (fBm) simulation

### 🚜 Refactor

- *(Simulation)* Improve code formatting and import organization
- *(circulant_embedding.rs)* Enhance performance and add variance normalization
- *(circulant_embedding.rs)* Remove variance normalization code
- *(utils)* Reorganize utility functions and add circulant embedding module

### ⚙️ Miscellaneous Tasks

- *(Dependencies)* Upgrade Rust dependencies for advanced numerical computing
- *(Dependencies)* Remove ndarray dependency from project

## [Rust-v0.1.4] - 2025-02-24

### 🚀 Features

- *(Rust)* Add occupation time for inverse subordinator
- *(Python)* Add functional simulation methods for first passage time and occupation time
- *(Python)* Add Poisson, Subordinator, and Inverse Subordinator simulation classes
- *(Rust)* Add Langevin equation simulation module
- *(Simulation)* Add Generalized Langevin Equation Simulation
- *(Simulation)* Add Subordinated Langevin Equation Simulation

### 🐛 Bug Fixes

- *(Langevin)* Correct stochastic simulation noise scaling

### 💼 Other

- Release Rust version 0.1.3
- Bump Python package version to 0.1.2

### 🚜 Refactor

- *(Langevin)* Remove unnecessary start position validation

### 📚 Documentation

- *(Rust)* Update CHANGELOG for version 0.1.3
- *(CHANGELOG)* Update changelog for Python version 0.1.2
- Update benchmark results with new hardware and software configuration
- *(Rust)* Update README and documentation for Langevin equation implementations

### ⚙️ Miscellaneous Tasks

- Update project dependencies and benchmark performance
- Bump package version to 0.1.4

## [Rust-0.1.3] - 2025-02-21

### 🚀 Features

- *(Rust)* Add subordinator simulation module
- *(Rust)* Add Poisson process simulation module
- *(Rust)* Add callable feature for simulation processes, which needs `nightly`.
- Add point process simulation methods for first passage and occupation time
- *(Rust)* Implement inverse subordinator simulation

### 🚜 Refactor

- Move simulate_with_duration implementation to traits module
- Standardize import statements and code formatting
- Optimize slice copying in point process duration simulation

### 📚 Documentation

- Improve documentation for Brownian motion and Lévy process simulations
- Update README with subordinator process roadmap
- *(Rust)* Implement subordinator and Poisson process simulations

### ⚙️ Miscellaneous Tasks

- Remove rust-toolchain.toml configuration
- Update Rust toolchain and license configuration
- Remove callable feature and nightly Rust toolchain

## [Rust-v0.1.2] - 2025-02-19

### 🚀 Features

- Add occupation time functionality
- Add Lévy process simulation and related functionality
- Add occupation time for Brownian motion

### 🐛 Bug Fixes

- *(Python)* Add input validation for Brownian motion and Lévy process methods

### 💼 Other

- Lower Python version requirement to 3.9

### 🚜 Refactor

- Remove gamma function implementations from utils
- Enhance simulation traits with continuous and point process abstractions
- Update Brownian motion and Lévy process simulation traits
- Optimize occupation time calculation using iterator methods
- Improve code formatting and import organization
- Introduce simulation prelude module for simplified imports
- Expose simulation module traits and functional components

### 📚 Documentation

- Add comprehensive README for Python package
- Update README with new features and progress
- Update project roadmap and feature tracking
- Update README files with comprehensive random number generation and simulation examples

### ⚙️ Miscellaneous Tasks

- Add PyPI publication workflow for Python package
- Switch Rust toolchain from stable to beta in publish workflow
- Remove test step from Python publish workflow
- Add changelog and git-cliff configuration
- Update CHANGELOG.md with recent project developments
- Bump project version to 0.1.2
- Prepare Rust release v0.1.2

## [0.1.0] - 2025-02-19

### 🚀 Features

- Add minmax utility function for finding min and max values in f64 arrays
- Add first passage time (FPT) calculation for Brownian motion
- Enhance first passage time (FPT) calculation with max duration
- Add Levy process simulation module

### 💼 Other

- Add justfile for generating Rust documentation

### 🚜 Refactor

- Traits
- Update Brownian motion simulation and traits
- Simplify Moment and Functional trait implementations
- Remove unused unchecked Brownian motion constructor
- Restructure simulation traits and add first passage time functionality
- Implement StochasticProcess and Trajectory for Brownian motion
- Remove nightly feature and related code

### 📚 Documentation

- Update README with comprehensive library overview and usage examples
- Simplify README with focused usage examples
- Update README with refined usage examples and syntax
- Update README with first passage time (FPT) example
- Refine README with updated Brownian motion simulation examples
- 中文,  English docs
- Update README with first passage time (FPT) max duration example
- Generate Rust documentation for DiffusionX library
- Update Rust documentation for DiffusionX library
- Modify justfile to clean documentation directory before generation
- Add documentation badge to README files
- Add Julia version reference to README files
- Update Rust documentation for simulation module
- Update README examples for Brownian motion simulation
- Add extensibility section to README files
- Minor README update for extensibility section
- Update documentation links in README files
- Update documentation badge links to Rust documentation path
- Minor documentation updates and formatting improvements
- Remove generated Rust documentation files

### 🧪 Testing

- Add comprehensive random number generation tests
- *(rust)* Add statistical utility functions for random number generation tests
- *(python)* Enhance stable distribution random number generation tests

### ⚙️ Miscellaneous Tasks

- Minor formatting in types module
- Add GitHub Pages deployment workflow for documentation

<!-- generated by git-cliff -->
