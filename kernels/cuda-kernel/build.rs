use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // List all CUDA kernel files
    let kernel_files = vec![
        "bm",
        "ou",
        "geometric_bm",
        "stats",
        "fbm",
        "cauchy",
        "gamma",
        "langevin",
        "levy",
        "levy_walk",
        "brownian_bridge",
        "brownian_excursion",
        "brownian_meander",
        "bng",
        "stable",
    ];

    // Watch for changes in kernel files
    for name in &kernel_files {
        println!("cargo:rerun-if-changed=src/{}.cu", name);
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");

    // Build PTX bindings for all kernels
    let mut builder = bindgen_cuda::Builder::default();

    for name in &kernel_files {
        let cu_file = format!("src/{}.cu", name);
        builder = builder.kernel(&cu_file);
    }

    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    println!(
        "cargo:info=Generated PTX bindings at {}",
        ptx_path.display()
    );
}
