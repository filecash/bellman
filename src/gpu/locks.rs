use fs2::FileExt;
use log::{debug, info, warn};
use std::fs::File;
use std::path::PathBuf;

// wdpost&wnpost parallel calc
use rust_gpu_tools::*;
use std::thread;
use std::time::Duration;

const GPU_LOCK_NAME: &str = "bellman.gpu.lock";
const PRIORITY_LOCK_NAME: &str = "bellman.priority.lock";
// wdpost&wnpost parallel calc
fn gpu_lock_path(filename: &str, bus_id: u32) -> PathBuf {
    let mut name = String::from(filename);
    name.push('.');
    name += &bus_id.to_string();
    let mut p = std::env::temp_dir();
    p.push(&name);
    p
}


/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[derive(Debug)]
// wdpost&wnpost parallel calc
pub struct GPULock(File, u32);
impl GPULock {
    pub fn id(&self) -> u32 {
        self.1
    }
    pub fn lock() -> GPULock {
        loop{
            let devs = rust_gpu_tools::opencl::Device::all();
            for dev in devs {
                let id = dev.bus_id().unwrap();
                let lock = gpu_lock_path(GPU_LOCK_NAME, id);
                let lock = File::create(&lock)
                    .unwrap_or_else(|_| panic!("Cannot create GPU lock file at {:?}", &lock));
                if lock.try_lock_exclusive().is_ok() {
                    return GPULock(lock, id);
                }
            }
            thread::sleep(Duration::from_secs(3));
        }
    }
}
impl Drop for GPULock {
    fn drop(&mut self) {
        debug!("GPU lock released!");
    }
}

/// `PrioriyLock` is like a flag. When acquired, it means a high-priority process
/// needs to acquire the GPU really soon. Acquiring the `PriorityLock` is like
/// signaling all other processes to release their `GPULock`s.
/// Only one process can have the `PriorityLock` at a time.
#[derive(Debug)]
// wdpost&wnpost parallel calc
pub struct PriorityLock(File, u32);
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        loop{
            let devs = opencl::Device::all();
            for dev in devs {
                let id = dev.bus_id().unwrap();
                let f = gpu_lock_path(PRIORITY_LOCK_NAME, id);
                let f = File::create(&f)
                    .unwrap_or_else(|_| panic!("Cannot create Priority lock file at {:?}", &f));
                if f.try_lock_exclusive().is_ok() {
                    return PriorityLock(f, id);
                }
            }
            thread::sleep(Duration::from_secs(3));
        }
    }
    pub fn wait(priority: bool) {
        if !priority {
            let _ = Self::lock();
        }
    }
    pub fn should_break(priority: bool) -> bool {
        !priority
            && {
                let mut r = true;
                let devs = opencl::Device::all();
                for dev in devs {
                    let id = dev.bus_id().unwrap();
                    let f = gpu_lock_path(PRIORITY_LOCK_NAME, id);
                    let f = File::create(&f)
                        .unwrap_or_else(|_| panic!("Cannot create Priority lock file at {:?}", &f));
                    if f.try_lock_exclusive().is_ok() {
                        r = false;
                        break;
                    }
                }
                r
            }
    }
}
impl Drop for PriorityLock {
    fn drop(&mut self) {
        debug!("Priority lock released!");
    }
}

use super::error::{GPUError, GPUResult};
use super::fft::FFTKernel;
use super::multiexp::MultiexpKernel;
use crate::bls::Engine;
use crate::domain::create_fft_kernel;
use crate::multiexp::create_multiexp_kernel;

macro_rules! locked_kernel {
    ($class:ident, $kern:ident, $func:ident, $name:expr) => {
        pub struct $class<E>
        where
            E: Engine,
        {
            log_d: usize,
            priority: bool,
            kernel: Option<$kern<E>>,
        }

        impl<E> $class<E>
        where
            E: Engine,
        {
            pub fn new(log_d: usize, priority: bool) -> $class<E> {
                $class::<E> {
                    log_d,
                    priority,
                    kernel: None,
                }
            }

            fn init(&mut self) {
                if self.kernel.is_none() {
                    PriorityLock::wait(self.priority);
                    info!("GPU is available for {}!", $name);
                    self.kernel = $func::<E>(self.log_d, self.priority);
                }
            }

            fn free(&mut self) {
                if let Some(_kernel) = self.kernel.take() {
                    warn!(
                        "GPU acquired by a high priority process! Freeing up {} kernels...",
                        $name
                    );
                }
            }

            pub fn with<F, R>(&mut self, mut f: F) -> GPUResult<R>
            where
                F: FnMut(&mut $kern<E>) -> GPUResult<R>,
            {
                if std::env::var("BELLMAN_NO_GPU").is_ok() {
                    return Err(GPUError::GPUDisabled);
                }

                self.init();

                loop {
                    if let Some(ref mut k) = self.kernel {
                        match f(k) {
                            Err(GPUError::GPUTaken) => {
                                self.free();
                                self.init();
                            }
                            Err(e) => {
                                warn!("GPU {} failed! Falling back to CPU... Error: {}", $name, e);
                                return Err(e);
                            }
                            Ok(v) => return Ok(v),
                        }
                    } else {
                        return Err(GPUError::KernelUninitialized);
                    }
                }
            }
        }
    };
}

locked_kernel!(LockedFFTKernel, FFTKernel, create_fft_kernel, "FFT");
locked_kernel!(
    LockedMultiexpKernel,
    MultiexpKernel,
    create_multiexp_kernel,
    "Multiexp"
);
