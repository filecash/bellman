use log::{info, warn};
use rust_gpu_tools::*;
use std::collections::HashMap;
use std::env;

#[derive(Copy, Clone)]
struct GPUInfo {
    core_count: usize,
    max_window_size: usize,
    chunk_size_scale: usize,
    best_chunk_size_scale: usize,
    reserved_mem_ratio: f32,
}

lazy_static::lazy_static! {
    static ref GPU_INFOS: HashMap<String, GPUInfo> = {
        let mut gpu_infos : HashMap<String, GPUInfo> = vec![
            // AMD
            ("gfx1010".to_string(), GPUInfo{core_count: 2560, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            // This value was chosen to give (approximately) empirically best performance for a Radeon Pro VII.
            ("gfx906".to_string(), GPUInfo{core_count: 7400, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),

            // NVIDIA
            ("Quadro RTX 6000".to_string(), GPUInfo{core_count: 4608, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),

            ("TITAN RTX".to_string(), GPUInfo{core_count: 4608, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),

            ("Tesla V100".to_string(), GPUInfo{core_count: 5120, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("Tesla P100".to_string(), GPUInfo{core_count: 3584, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("Tesla T4".to_string(), GPUInfo{core_count: 2560, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("Quadro M5000".to_string(), GPUInfo{core_count: 2048, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),

            ("GeForce RTX 3090".to_string(), GPUInfo{core_count: 10496, max_window_size: 9, chunk_size_scale: 90000, best_chunk_size_scale: 90000, reserved_mem_ratio: 0.05}),
            ("GeForce RTX 3080".to_string(), GPUInfo{core_count: 8704, max_window_size: 8, chunk_size_scale: 2900, best_chunk_size_scale: 29, reserved_mem_ratio: 0.2}),
            ("GeForce RTX 3070".to_string(), GPUInfo{core_count: 5888, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),

            // ("GeForce RTX 2080 Ti".to_string(), GPUInfo{core_count: 8704, max_window_size: 8, chunk_size_scale: 92, best_chunk_size_scale: 92, reserved_mem_ratio: 0.2}),
            ("GeForce RTX 2080 Ti".to_string(), GPUInfo{core_count: 4352, max_window_size: 9, chunk_size_scale: 29, best_chunk_size_scale: 29, reserved_mem_ratio: 0.2}),

            ("GeForce RTX 2080 SUPER".to_string(), GPUInfo{core_count: 3072, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("GeForce RTX 2080".to_string(), GPUInfo{core_count: 2944, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("GeForce RTX 2070 SUPER".to_string(), GPUInfo{core_count: 2560, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),

            ("GeForce GTX 1080 Ti".to_string(), GPUInfo{core_count: 3584, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1080".to_string(), GPUInfo{core_count: 2560, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 2060".to_string(), GPUInfo{core_count: 1920, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1660 Ti".to_string(), GPUInfo{core_count: 1536, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1060".to_string(), GPUInfo{core_count: 1280, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1650 SUPER".to_string(), GPUInfo{core_count: 1280, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1650".to_string(), GPUInfo{core_count: 896, max_window_size: 0, chunk_size_scale: 0, best_chunk_size_scale: 0, reserved_mem_ratio: 0.2}),
        ].into_iter().collect();

        match env::var("BELLMAN_CUSTOM_GPU").and_then(|var| {
            for card in var.split(",") {
                let splitted = card.split(":").collect::<Vec<_>>();
                let name = splitted[0].trim().to_string();

                let cores: usize = if 2 <= splitted.len() {
                    splitted[1].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!")
                } else {
                    DEFAULT_CORE_COUNT
                };
                let max_window_size: usize = if 3 <= splitted.len() {
                    splitted[2].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!")
                } else {
                    10
                };
                let chunk_size_scale: usize = if 4 <= splitted.len() {
                    splitted[3].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!")
                } else {
                    2
                };
                let best_chunk_size_scale: usize = if 5 <= splitted.len() {
                    splitted[4].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!")
                } else {
                    2
                };
                let reserved_mem_ratio : f32 = if 6 <= splitted.len() {
                    splitted[5].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!")
                } else {
                    0.2
                };
                info!("Adding \"{}\" to GPU list {} CUDA cores {} max window size {} chunk size scale {} best chunk size scale {} reserved mem ratio.",
                      name, cores, max_window_size, chunk_size_scale, best_chunk_size_scale, reserved_mem_ratio);
                gpu_infos.insert(name,
                                 GPUInfo{
                                     core_count: cores,
                                     max_window_size,
                                     chunk_size_scale,
                                     best_chunk_size_scale,
                                     reserved_mem_ratio,
                                 });
            }
            Ok(())
        }) { Err(_) => { }, Ok(_) => { } }

        gpu_infos
    };
}

const DEFAULT_CORE_COUNT: usize = 2560;
pub fn get_core_count(d: &opencl::Device) -> usize {
    let name = d.name();
    match GPU_INFOS.get(&name[..]) {
        Some(&info) => info.core_count,
        None => {
            warn!(
                "Number of CUDA cores for your device ({}) is unknown! Best performance is \
                 only achieved when the number of CUDA cores is known! You can find the \
                 instructions on how to support custom GPUs here: \
                 https://lotu.sh/en+hardware-mining",
                name
            );
            DEFAULT_CORE_COUNT
        }
    }
}

pub fn get_max_window_size(d: &opencl::Device) -> usize {
    let name = d.name();
    match GPU_INFOS.get(&name[..]) {
        Some(&info) => info.max_window_size,
        None => 10,
    }
}

pub fn get_chunk_size_scale(d: &opencl::Device) -> usize {
    let name = d.name();
    match GPU_INFOS.get(&name[..]) {
        Some(&info) => info.chunk_size_scale,
        None => 2,
    }
}

pub fn get_best_chunk_size_scale(d: &opencl::Device) -> usize {
    let name = d.name();
    match GPU_INFOS.get(&name[..]) {
        Some(&info) => info.best_chunk_size_scale,
        None => 2,
    }
}

pub fn get_reserved_mem_ratio(d: &opencl::Device) -> f32 {
    let name = d.name();
    match GPU_INFOS.get(&name[..]) {
        Some(&info) => info.reserved_mem_ratio,
        None => 0.2,
    }
}

pub fn dump_device_list() {
    for d in opencl::Device::all() {
        info!("Device: {:?}", d);
    }
}

#[cfg(feature = "gpu")]
#[test]
pub fn test_list_devices() {
    let _ = env_logger::try_init();
    dump_device_list();
}
