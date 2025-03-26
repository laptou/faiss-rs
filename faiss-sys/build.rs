fn main() {
    #[cfg(feature = "static-faiss")]
    static_link_faiss();

    #[cfg(not(feature = "static-faiss"))]
    println!("cargo:rustc-link-lib=faiss_c");
}

#[cfg(feature = "static-faiss")]
fn static_link_faiss() {
    use std::{ffi::OsString, path::PathBuf, process::Command};

    let mut cfg = cmake::Config::new("faiss");

    cfg.define("FAISS_ENABLE_C_API", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define(
            "FAISS_ENABLE_GPU",
            if cfg!(feature = "gpu") { "ON" } else { "OFF" },
        )
        .define("FAISS_ENABLE_PYTHON", "OFF")
        .define("BUILD_TESTING", "OFF")
        .profile("RelWithDebInfo")
        .very_verbose(true);

    let blas_link_mode = if cfg!(feature = "static-blas") {
        "static"
    } else {
        "dylib"
    };

    let omp_link_mode = if cfg!(feature = "static-omp") {
        "static"
    } else {
        "dylib"
    };

    if let Ok(oneapi_root) = std::env::var("ONEAPI_ROOT") {
        println!("using ONEAPI_ROOT={oneapi_root:?}");

        // if using Intel oneAPI (such as on Windows), we need libraries from
        // the mkl folder and the compiler folder
        let oneapi_root = PathBuf::from(oneapi_root);
        let mkl_root = oneapi_root.join("mkl").join("latest");
        let compiler_root = oneapi_root.join("compiler").join("latest");

        println!(
            "cargo:rustc-link-search=native={}",
            mkl_root.join("bin").display()
        );
        println!("cargo:rustc-link-search={}", mkl_root.join("lib").display());

        println!(
            "cargo:rustc-link-search=native={}",
            compiler_root.join("bin").display()
        );
        println!(
            "cargo:rustc-link-search={}",
            compiler_root.join("lib").display()
        );

        #[cfg(windows)]
        {
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_intel_lp64_dll");
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_intel_thread_dll");
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_core_dll");
        }

        #[cfg(unix)]
        {
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_intel_lp64");
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_intel_thread");
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_core");
        }

        println!("cargo:rustc-link-lib={omp_link_mode}=libiomp5md");

        cfg.env("MKLROOT", &mkl_root);
        cfg.env("LIB", mkl_root.join("lib"));
    } else if let Ok(mkl_root) = std::env::var("MKLROOT") {
        println!("using MKLROOT={mkl_root:?}");
        let mkl_root = PathBuf::from(mkl_root);

        println!(
            "cargo:rustc-link-search=native={}",
            mkl_root.join("bin").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            mkl_root.join("lib").display()
        );

        #[cfg(windows)]
        {
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_intel_lp64_dll");
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_intel_thread_dll");
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_core_dll");
        }

        #[cfg(unix)]
        {
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_intel_lp64");
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_intel_thread");
            println!("cargo:rustc-link-lib={blas_link_mode}=mkl_core");
        }

        println!("cargo:rustc-link-lib={omp_link_mode}=libiomp5md");
        cfg.define("MKLROOT", mkl_root);
    } else {
        println!("cargo:rustc-link-lib={blas_link_mode}=blas");
        println!("cargo:rustc-link-lib={blas_link_mode}=lapack");

        #[cfg(target_os = "macos")]
        {
            // on macOS, libomp is often installed via Homebrew; linking against it
            // requires special handling

            if let Ok(openmp_prefix) = Command::new("brew")
                .args(&["--prefix", "libomp"])
                .output()
                .map_err(|e| format!("Failed to run brew: {}", e))
            {
                let openmp_prefix =
                    PathBuf::from(String::from_utf8(openmp_prefix.stdout).unwrap().trim());
                let openmp_lib_path = openmp_prefix.join("lib");

                println!(
                    "cargo:rustc-link-search=native={}",
                    openmp_lib_path.display()
                );

                cfg.define("OpenMP_omp_LIBRARY", openmp_lib_path.join("libomp.dylib"));
                cfg.define("OpenMP_CXX_LIB_NAMES", "omp");
                cfg.define(
                    "OpenMP_CXX_FLAGS",
                    format!("-I{}", openmp_prefix.join("include").display()),
                );
            }

            println!("cargo:rustc-link-lib={omp_link_mode}=omp");
        }

        #[cfg(not(target_os = "macos"))]
        {
            println!("cargo:rustc-link-lib={blas_link_mode}=gomp");
        }
    }

    println!("cargo:rerun-if-env-changed=ONEAPI_ROOT");
    println!("cargo:rerun-if-env-changed=MKLROOT");

    let profile = cfg.get_profile().to_owned();
    let dst = cfg.build();

    let faiss_location = dst.join("lib");

    // CMake on Windows puts the C API library in a subfolder based on the build
    // profile
    #[cfg(windows)]
    let faiss_c_location = dst.join("build\\c_api").join(profile);

    #[cfg(not(windows))]
    let faiss_c_location = dst.join("build/c_api");

    println!(
        "cargo:rustc-link-search=native={}",
        faiss_location.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        faiss_c_location.display()
    );
    println!("cargo:rustc-link-lib=static=faiss_c");
    println!("cargo:rustc-link-lib=static=faiss");

    link_cxx();

    if cfg!(feature = "gpu") {
        let cuda_path = cuda_lib_path();
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }
}

#[cfg(feature = "static")]
fn link_cxx() {
    let cxx = match std::env::var("CXXSTDLIB") {
        Ok(s) if s.is_empty() => None,
        Ok(s) => Some(s),
        Err(_) => {
            let target = std::env::var("TARGET").unwrap();
            if target.contains("msvc") {
                None
            } else if target.contains("apple")
                | target.contains("freebsd")
                | target.contains("openbsd")
            {
                Some("c++".to_string())
            } else {
                Some("stdc++".to_string())
            }
        }
    };
    if let Some(cxx) = cxx {
        println!("cargo:rustc-link-lib={}", cxx);
    }
}

#[cfg(feature = "static")]
fn cuda_lib_path() -> String {
    // look for CUDA_PATH in environment,
    // then CUDA_LIB_PATH,
    // then CUDA_INCLUDE_PATH
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        return cuda_path;
    }
    if let Ok(cuda_lib_path) = std::env::var("CUDA_LIB_PATH") {
        return cuda_lib_path;
    }
    if let Ok(cuda_include_path) = std::env::var("CUDA_INCLUDE_PATH") {
        return cuda_include_path;
    }

    panic!("Could not find CUDA: environment variables `CUDA_PATH`, `CUDA_LIB_PATH`, or `CUDA_INCLUDE_PATH` must be set");
}
