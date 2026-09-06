fn main() {
    #[cfg(feature = "gpu")]
    build_cuda_libs();
}

#[cfg(feature = "gpu")]
fn build_cuda_libs() {
    println!("cargo::rerun-if-changed=kernels/");

    println!("cargo:rerun-if-env-changed=VANITY_CUDA_ARCH");
    let mut build = cc::Build::new();
    println!("cargo:rerun-if-env-changed=VANITY_EVM_BATCH_SIZE");
    if let Ok(size) = std::env::var("VANITY_EVM_BATCH_SIZE") {
        assert!(
            ["8", "16", "32"].contains(&size.as_str()),
            "VANITY_EVM_BATCH_SIZE must be 8, 16 or 32"
        );
        build.flag(format!("-DSECP256K1_BATCH_SIZE={size}"));
    }

    if let Ok(arch) = std::env::var("VANITY_CUDA_ARCH") {
        assert!(
            arch.bytes().all(|c| c.is_ascii_digit()) && !arch.is_empty(),
            "VANITY_CUDA_ARCH must be numeric, e.g. 89"
        );
        build.flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        build.flag(format!("-gencode=arch=compute_{arch},code=compute_{arch}"));
    } else {
        for arch in [75, 80, 86, 89] {
            build.flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
        build.flag("-gencode=arch=compute_86,code=compute_86");
    }
    build
        .cuda(true)
        // Core infrastructure
        .file("kernels/cuda_init.cu")
        .file("kernels/utils.cu")
        .file("kernels/gpu_config.cu") // GPU auto-detection and configuration
        // Solana dependencies
        .file("kernels/base58_simple.cu") // Base58 encoding for Solana addresses
        .file("kernels/sha256.cu") // SHA-256 for Ed25519
        .file("kernels/ed25519/fe.cu") // Field elements
        .file("kernels/ed25519/ge.cu") // Group elements
        .file("kernels/ed25519/sc.cu") // Scalar operations
        .file("kernels/ed25519/keypair.cu") // Ed25519 keypair generation
        .file("kernels/ed25519/sha512.cu") // SHA-512 for Ed25519
        .file("kernels/ed25519/seed.cu") // Seed handling
        .file("kernels/sol_vanity_optimized.cu") // Optimized Solana kernel
        // Ethereum dependencies
        .file("kernels/keccak256.cu") // Keccak256 hashing
        .file("kernels/secp256k1_mrspike.cu") // secp256k1 curve operations
        .file("kernels/eth_vanity_ultra.cu") // Ultra-optimized auto-scaling Ethereum kernel
        .include("kernels")
        .include("kernels/ed25519")
        .flag("-cudart=static")
        .flag("--expt-relaxed-constexpr")
        .flag("-maxrregcount=0") // Allow unlimited registers for complex kernels
        .compile("libvanity.a");

    // Add link directory and required libraries
    #[cfg(target_os = "windows")]
    {
        // Prefer CUDA_PATH env var if available; fallback to a common default.
        let cuda_path = std::env::var("CUDA_PATH").unwrap_or_else(|_| {
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0".to_string()
        });
        println!(
            "cargo:rustc-link-search=native={}",
            format!("{}\\lib\\x64", cuda_path)
        );
        // CUDA driver library (must be linked even with static runtime)
        println!("cargo:rustc-link-lib=cuda");
        // Windows system libraries required by CUDA static runtime
        println!("cargo:rustc-link-lib=dylib=ole32");
        println!("cargo:rustc-link-lib=dylib=oleaut32");
    }
    #[cfg(not(target_os = "windows"))]
    {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
    }

    // Emit the location of the compiled library
    let out_dir = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", out_dir);
}
