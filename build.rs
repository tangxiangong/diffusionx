#[cfg(feature = "cuda")]
use std::{env, path::PathBuf, process::Command};

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn main() {}

#[cfg(all(feature = "cuda", feature = "metal"))]
fn main() {
    panic!("Cannot enable both CUDA and Metal features");
}

#[cfg(feature = "cuda")]
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
