use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let kernels = ["bm", "stable", "levy"];

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    for kernel in kernels {
        let kernel_file = format!("src/{kernel}.cu");
        println!("cargo:rerun-if-changed={kernel_file}");

        let ptx_path = out_dir.join(&format!("{kernel}.ptx"));

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
