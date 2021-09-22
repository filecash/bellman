use super::error::{GPUError, GPUResult};
use super::locks;
use super::sources;
use super::utils;
use crate::bls::Engine;
use crate::multicore::Worker;
use crate::multiexp::{multiexp as cpu_multiexp, FullDensity};
use ff::{PrimeField, ScalarEngine};
use groupy::{CurveAffine, CurveProjective};
use log::{error, info};
use rayon::prelude::*;
use rust_gpu_tools::*;
use std::any::TypeId;
use std::sync::Arc;

// Added by jackoelv for C2 20210330
use std::env;
use std::sync::mpsc;
extern crate scoped_threadpool;
use scoped_threadpool::Pool;

const LOCAL_WORK_SIZE: usize = 256;
// const MAX_WINDOW_SIZE: usize = 10;
// const MEMORY_PADDING: f64 = 0.2f64; // Let 20% of GPU memory be free

pub fn get_cpu_utilization() -> f64 {
    use std::env;
    env::var("BELLMAN_CPU_UTILIZATION")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid BELLMAN_CPU_UTILIZATION! Defaulting to 0...");
                Ok(0f64)
            }
        })
        .unwrap_or(0f64)
        .max(0f64)
        .min(1f64)
}

// Multiexp kernel for a single GPU
pub struct SingleMultiexpKernel<E>
where
    E: Engine,
{
    gpu_id: u32,
    max_window_size: usize,
    chunk_size_scale: usize,
    best_chunk_size_scale: usize,
    reserved_mem_ratio: f32,

    program: opencl::Program,

    core_count: usize,
    n: usize,

    priority: bool,
    _phantom: std::marker::PhantomData<E::Fr>,
}

fn calc_num_groups(core_count: usize, num_windows: usize) -> usize {
    // Observations show that we get the best performance when num_groups * num_windows ~= 2 * CUDA_CORES
    2 * core_count / num_windows
}

// Deleted by jackoelv for C2 20210330
fn calc_window_size(n: usize, exp_bits: usize, core_count: usize, max_window_size: usize) -> usize {
//     // window_size = ln(n / num_groups)
//     // num_windows = exp_bits / window_size
//     // num_groups = 2 * core_count / num_windows = 2 * core_count * window_size / exp_bits
//     // window_size = ln(n / num_groups) = ln(n * exp_bits / (2 * core_count * window_size))
//     // window_size = ln(exp_bits * n / (2 * core_count)) - ln(window_size)
//     //
//     // Thus we need to solve the following equation:
//     // window_size + ln(window_size) = ln(exp_bits * n / (2 * core_count))
//     let lower_bound = (((exp_bits * n) as f64) / ((2 * core_count) as f64)).ln();
//     for w in 0..MAX_WINDOW_SIZE {
//         if (w as f64) + (w as f64).ln() > lower_bound {
//             return w;
//         }
//     }
//    MAX_WINDOW_SIZE
    max_window_size
}

fn calc_best_chunk_size(max_window_size: usize, core_count: usize, exp_bits: usize, scale: usize) -> usize {
    // Best chunk-size (N) can also be calculated using the same logic as calc_window_size:
    // n = e^window_size * window_size * 2 * core_count / exp_bits
    info!("calc_best_chunk_size = {}",  (((max_window_size as f64).exp() as f64)
    * (max_window_size as f64)
    * scale as f64
    * (core_count as f64)
    / (exp_bits as f64))
    .ceil() as usize);

    (((max_window_size as f64).exp() as f64)
        * (max_window_size as f64)

        * scale as f64  //* 2f64

        * (core_count as f64)
        / (exp_bits as f64))
        .ceil() as usize
}

fn calc_chunk_size<E>(mem: u64, core_count: usize, scale: usize, max_window_size: usize, reserved_mem_ratio: f32) -> usize
where
    E: Engine,
{
    let aff_size = std::mem::size_of::<E::G1Affine>() + std::mem::size_of::<E::G2Affine>();
    let exp_size = exp_size::<E>();
    let proj_size = std::mem::size_of::<E::G1>() + std::mem::size_of::<E::G2>();

//    ((((mem as f64) * (1f64 - MEMORY_PADDING)) as usize)
//        - (2 * core_count * ((1 << MAX_WINDOW_SIZE) + 1) * proj_size))
//        / (aff_size + exp_size)
    info!("exp_size = {}   aff_size =  {}", exp_size, aff_size);
    info!(" calc_chunk_size = {}", ((((mem as f64) * (1f64 - reserved_mem_ratio as f64)) as usize)
    - (scale * core_count * ((1 << max_window_size) + 1) * proj_size))
    / (aff_size + exp_size));
    ((((mem as f64) * (1f64 - reserved_mem_ratio as f64)) as usize)
        - (scale * core_count * ((1 << max_window_size) + 1) * proj_size))
        / (aff_size + exp_size)
}

fn exp_size<E: Engine>() -> usize {
    std::mem::size_of::<<E::Fr as ff::PrimeField>::Repr>()
}

impl<E> SingleMultiexpKernel<E>
where
    E: Engine,
{
    // Modified by jackoelv for C2 20210330
    pub fn create(d: opencl::Device, priority: bool, gpuid: u32) -> GPUResult<SingleMultiexpKernel<E>> {
        let src = sources::kernel::<E>(d.brand() == opencl::Brand::Nvidia);

        let exp_bits = exp_size::<E>() * 8;
        let core_count = utils::get_core_count(&d);
        let mem = d.memory();
        //let max_n = calc_chunk_size::<E>(mem, core_count);
        //let best_n = calc_best_chunk_size(MAX_WINDOW_SIZE, core_count, exp_bits);
        let reserved_mem_ratio = utils::get_reserved_mem_ratio(&d);
        let max_window_size = utils::get_max_window_size(&d);
        let chunk_size_scale = utils::get_chunk_size_scale(&d);
        let best_chunk_size_scale = utils::get_best_chunk_size_scale(&d);
        let max_n = calc_chunk_size::<E>(mem, core_count, chunk_size_scale, max_window_size, reserved_mem_ratio);
        let best_n = calc_best_chunk_size(max_window_size, core_count, exp_bits, best_chunk_size_scale);

        let n = std::cmp::min(max_n, best_n);

        Ok(SingleMultiexpKernel {
            gpu_id: gpuid,
            max_window_size,
            chunk_size_scale,
            best_chunk_size_scale,
            reserved_mem_ratio,

            program: opencl::Program::from_opencl(d, &src)?,
            core_count,
            n,
            priority,
            _phantom: std::marker::PhantomData,
        })
    }

    // Modified by jackoelv for C2 20210330
    pub fn multiexp<G>(
        &mut self,
        bases: &[G],
        exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
        n: usize,
        //jack_windows_size: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
    {
        // Deleted by long 20210816
        // if locks::PriorityLock::should_break(self.priority) {
        //     return Err(GPUError::GPUTaken);
        // }

        let exp_bits = exp_size::<E>() * 8;

        //let window_size = calc_window_size(n as usize, exp_bits, self.core_count);
        //let window_size = jack_windows_size;
        let window_size = calc_window_size(n as usize, exp_bits, self.core_count, self.max_window_size);

        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = calc_num_groups(self.core_count, num_windows);
        let bucket_len = 1 << window_size;

        // info!("bucket_len is :{}",  bucket_len);

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let mut base_buffer = self.program.create_buffer::<G>(n)?;
        base_buffer.write_from(0, bases)?;
        let mut exp_buffer = self.program.create_buffer::<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>(n)?;
        exp_buffer.write_from(0, exps)?;

        let bucket_buffer = self.program.create_buffer::<<G as CurveAffine>::Projective>((4 * self.core_count * bucket_len) as usize)?;
        let result_buffer = self.program.create_buffer::<<G as CurveAffine>::Projective>((4 * self.core_count) as usize)?;

        // Make global work size divisible by `LOCAL_WORK_SIZE`
        let mut global_work_size = num_windows * num_groups;
        global_work_size +=
            (LOCAL_WORK_SIZE - (global_work_size % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE;

        let kernel = self.program.create_kernel(
            if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                "G1_bellman_multiexp"
            } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
                "G2_bellman_multiexp"
            } else {
                return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
            },
            global_work_size,
            None,
        );

        kernel
            .arg(&base_buffer)
            .arg(&bucket_buffer)
            .arg(&result_buffer)
            .arg(&exp_buffer)
            .arg(n as u32)
            .arg(num_groups as u32)
            .arg(num_windows as u32)
            .arg(window_size as u32)
            .run()?;

        let mut results = vec![<G as CurveAffine>::Projective::zero(); num_groups * num_windows];
        result_buffer.read_into(0, &mut results)?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as CurveAffine>::Projective::zero();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }
}

// A struct that containts several multiexp kernels for different devices
pub struct MultiexpKernel<E>
where
    E: Engine,
{
    kernels: Vec<SingleMultiexpKernel<E>>,
    //_lock: locks::GPULock, // RFC 1857: struct fields are dropped in the same order as they are declared.
    _lock: Vec<locks::GPULock>, // RFC 1857: struct fields are dropped in the same order as they are declared.
}

impl<E> MultiexpKernel<E>
where
    E: Engine,
{
    pub fn create(priority: bool) -> GPUResult<MultiexpKernel<E>> {
        let lock = locks::GPULock::lock_all(); // Modified by long 20210512
        // wdpost&wnpost parallel calc
        let id = lock.id();

        let devices = opencl::Device::all();
        let locks = vec![lock];

        let kernels: Vec<_> = devices
            .into_iter()
            // wdpost&wnpost parallel calc
            .filter(| d | d.bus_id().unwrap() ==  id )

            //.map(|d| (d.clone(), SingleMultiexpKernel::<E>::create(d.clone(), priority)))
            .map(|d| (d, SingleMultiexpKernel::<E>::create(d.clone(), priority, id)))
            .filter_map(|(device, res)| {
                if let Err(ref e) = res {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                res.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }
        info!(
            "Multiexp: {} working device(s) selected. (CPU utilization: {})",
            kernels.len(),
            get_cpu_utilization()
        );
        // wdpost&wnpost parallel calc
        //for (i, k) in kernels.iter().enumerate() {
        for (_, k) in kernels.iter().enumerate() {
            info!(
                "Multiexp: Device {}: {} (Chunk-size: {})",
                // wdpost&wnpost parallel calc
                k.program.device().bus_id().unwrap(), //i,
                
                k.program.device().name(),
                k.n
            );
        }
        Ok(MultiexpKernel::<E> {
            kernels,
            _lock: locks,
        })
    }

    pub fn sched_multiexp(priority: bool) -> GPUResult<MultiexpKernel<E>> {
        let devices = opencl::Device::all();
        if devices.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }

        let max_gpu_num = env::var("BELLMAN_GPU_NUM_PER_MULTIEXP").and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid BELLMAN_GPU_NUM_PER_MULTIEXP! Defaulting to 1...");
                Ok(1_usize)
            }
        })
        .unwrap_or(1_usize);

        let mut kernels: Vec<_> = vec![];
        let mut kernel_ids: Vec<_> = vec![];
        let mut locks: Vec<locks::GPULock> = vec![];
        for i in 0..devices.len() {
            if kernels.len() == max_gpu_num {
                info!("out the range of max gpu num per multiexp({}), stop create kernel", max_gpu_num);
                break;
            }

            let res = locks::GPULock::try_lock(i as u32);
            match res {
                Ok(lk) => {
                    let resl = SingleMultiexpKernel::<E>::create(devices[i].clone(), priority, i as u32);
                    match resl {
                        Ok(k) => {
                            kernels.push(k);
                            kernel_ids.push(i);
                            locks.push(lk);
                        },
                        Err(e) => {
                            error!(
                                "Cannot initialize kernel for device '({}: {})'! Error: {}",
                                i,
                                devices[i].name(),
                                e
                            );
                        }
                    }
                },
                Err(_) => {},
            }
        }

        if kernels.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }
        info!(
            "Multiexp: {} working device(s) selected. (CPU utilization: {})",
            kernels.len(),
            get_cpu_utilization()
        );
        for (i, k) in kernels.iter().enumerate() {
            info!(
                "Multiexp: Device selected ({}: {}) (Chunk-size: {})",
                kernel_ids[i],
                k.program.device().name(),
                k.n
            );
        }
        return Ok(MultiexpKernel::<E> {
            kernels,
            _lock: locks,
        })
    }
    
    // Modified by jackoelv for C2 20210330
    pub fn multiexp<G>(
        &mut self,
        pool: &Worker,
        bases: Arc<Vec<G>>,
        exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
        skip: usize,
        n: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
        <G as groupy::CurveAffine>::Engine: crate::bls::Engine,
    {
        let num_devices = self.kernels.len();
        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases[skip..(skip + n)];
        let exps = &exps[..n];

        let cpu_n = ((n as f64) * get_cpu_utilization()) as usize;
        let n = n - cpu_n;
        let (cpu_bases, bases) = bases.split_at(cpu_n);
        let (cpu_exps, exps) = exps.split_at(cpu_n);

        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;

//         crate::multicore::THREAD_POOL.install(|| {
//             use rayon::prelude::*;
//             let mut acc = <G as CurveAffine>::Projective::zero();
//             let results = if n > 0 {
//                 bases
//                     .par_chunks(chunk_size)
//                     .zip(exps.par_chunks(chunk_size))
//                     .zip(self.kernels.par_iter_mut())
//                     .map(|((bases, exps), kern)| -> Result<<G as CurveAffine>::Projective, GPUError> {
//                         let mut acc = <G as CurveAffine>::Projective::zero();
//                         for (bases, exps) in bases.chunks(kern.n).zip(exps.chunks(kern.n)) {
//                             let result = kern.multiexp(bases, exps, bases.len())?;
//                             acc.add_assign(&result);
//                         }
//                         Ok(acc)
//                     })
//                     .collect::<Vec<_>>()
//             } else {
//                 Vec::new()
//             };
//             let cpu_acc = cpu_multiexp(
//                 &pool,
//                 (Arc::new(cpu_bases.to_vec()), 0),
//                 FullDensity,
//                 Arc::new(cpu_exps.to_vec()),
//                 &mut None,
//             );
//             for r in results {
//                 acc.add_assign(&r?);
//             }
//             acc.add_assign(&cpu_acc.wait().unwrap());
//             Ok(acc)
//         })
//     }
// }

        crate::multicore::THREAD_POOL.install(|| {
            use rayon::prelude::*;
            let mut acc = <G as CurveAffine>::Projective::zero();

            // concurrent computing
            let (tx_gpu, rx_gpu) = mpsc::channel();
            let (tx_cpu, rx_cpu) = mpsc::channel();
            let mut scoped_pool = Pool::new(2);
            scoped_pool.scoped(|scoped| {
                // GPU
                scoped.execute(move || {
                    let results = if n > 0 {
                        bases
                            .par_chunks(chunk_size)
                            .zip(exps.par_chunks(chunk_size))
                            .zip(self.kernels.par_iter_mut())
                            .map(|((bases, exps), kern)| -> Result<<G as CurveAffine>::Projective, GPUError> {
                                let mut acc = <G as CurveAffine>::Projective::zero();
	                            let mut jack_chunk = kern.n;
                                let size_result = std::mem::size_of::<<G as CurveAffine>::Projective>();
                                //let mut jack_windows_size = 11;
                                // if size_result > 144 {
                                //     jack_windows_size = 8;
                                // }
                                for (bases, exps) in bases.chunks(jack_chunk).zip(exps.chunks(jack_chunk)) {
                                    //let result = kern.multiexp(bases, exps, bases.len(), jack_windows_size)?;
                                    let result = kern.multiexp(bases, exps, bases.len())?;
                                    acc.add_assign(&result);
                                }

                                Ok(acc)
                            })
                            .collect::<Vec<_>>()
                    } else {
                        Vec::new()
                    };

                    tx_gpu.send(results).unwrap();

                });
                // CPU
                scoped.execute(move || {
                    let cpu_acc = cpu_multiexp(
                        &pool,
                        (Arc::new(cpu_bases.to_vec()), 0),
                        FullDensity,
                        Arc::new(cpu_exps.to_vec()),
                        &mut None,
                    );
                    let cpu_r = cpu_acc.wait().unwrap();

                    tx_cpu.send(cpu_r).unwrap();
                });
            });

            // waiting results...
            let results = rx_gpu.recv().unwrap();
            let cpu_r = rx_cpu.recv().unwrap();

            for r in results {
                match r {
                    Ok(r) => acc.add_assign(&r),
                    Err(e) => return Err(e),
                }
            }

            // acc.add_assign(&cpu_acc.wait().unwrap());
            acc.add_assign(&cpu_r);

            Ok(acc)
        })
    }
}
