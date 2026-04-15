#[cfg(any(
    all(feature = "cuda", not(feature = "metal")),
    all(feature = "metal", not(feature = "cuda"))
))]
use std::{env, path::PathBuf, process::Command};

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn main() {}

#[cfg(all(feature = "cuda", feature = "metal"))]
fn main() {
    panic!("Cannot enable both CUDA and Metal features");
}

#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let kernels = ["bm", "random", "levy", "ou"];

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    for kernel in kernels {
        let kernel_file = format!("kernels/cuda-kernel/{kernel}.cu");
        println!("cargo:rerun-if-changed={kernel_file}");

        let ptx_path = out_dir.join(format!("{kernel}.ptx"));

        let status = Command::new("nvcc")
            .args([
                "-ptx",
                &kernel_file,
                "-o",
                ptx_path.to_str().expect("Invalid PTX path"),
            ])
            .status()
            .expect("Failed to execute nvcc. Is CUDA installed and nvcc in PATH?");

        if !status.success() {
            panic!("nvcc failed with status: {status}");
        }

        let upper_kernel = kernel.to_uppercase();

        println!(
            "cargo:rustc-env={upper_kernel}_KERNEL_PTX={}",
            ptx_path.display()
        );
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let kernels = ["bm", "levy", "ou", "random"];
    let kernel_dir = "kernels/metal-kernel";

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // Read utils.metal content for inclusion
    let utils_path = format!("{kernel_dir}/utils.metal");
    println!("cargo:rerun-if-changed={utils_path}");

    for kernel in kernels {
        let kernel_file = format!("{kernel_dir}/{kernel}.metal");
        println!("cargo:rerun-if-changed={kernel_file}");

        let metallib_path = out_dir.join(format!("{kernel}.metallib"));
        let air_path = out_dir.join(format!("{kernel}.air"));

        // Compile .metal to .air (Apple Intermediate Representation)
        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                &kernel_file,
                "-I",
                kernel_dir, // Include directory for utils.metal
                "-o",
                air_path.to_str().expect("Invalid AIR path"),
            ])
            .status()
            .expect("Failed to execute xcrun metal. Is Xcode installed?");

        if !status.success() {
            panic!("Metal compiler failed with status: {status}");
        }

        // Link .air to .metallib
        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metallib",
                air_path.to_str().expect("Invalid AIR path"),
                "-o",
                metallib_path.to_str().expect("Invalid metallib path"),
            ])
            .status()
            .expect("Failed to execute xcrun metallib");

        if !status.success() {
            panic!("Metal linker failed with status: {status}");
        }

        let upper_kernel = kernel.to_uppercase();

        println!(
            "cargo:rustc-env={upper_kernel}_KERNEL_METALLIB={}",
            metallib_path.display()
        );
    }
}
