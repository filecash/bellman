#![allow(unused_imports)]
#![allow(unused_variables)]
extern crate bellperson;
extern crate paired;
extern crate rand;
extern crate ff;

#[macro_use]
extern crate log;

use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use bellperson::groth16::{Parameters};
use paired::{Engine};
use ff::{Field, PrimeField};

use std::fs::File;
use std::io::prelude::*;

use std::time::{Duration, Instant};
use std::thread::sleep;

mod dummy;

fn main(){
    use paired::bls12_381::{Bls12, Fr};
    use rand::thread_rng;
    use bellperson::groth16::{
        create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof, Proof,
    };

    pretty_env_logger::try_init().unwrap();
    info!("I know the value of 2^(2^1000)");

    let rng = &mut thread_rng();

    let args = std::env::args().collect::<Vec<String>>();
    let load_parameters = &args.iter().nth(1).cloned().unwrap_or_default() == "load";
    let parameters_path = "parameters-1.dat";

    // Create parameters for our circuit
    let params = if load_parameters {
        info!("loading parameters...");
        let param_file = File::open(parameters_path).expect("Unable to open parameters file!");
        Parameters::<Bls12>::read(param_file, false /* false for better performance*/)
            .expect("Unable to read parameters file!")
    } else {
        info!("creating parameters...");
        let c = dummy::DummyDemo::<Bls12> {
            xx: None
        };

        let p = generate_random_parameters(c, rng).unwrap();
        let param_file = File::create(parameters_path).expect("Unable to create parameters file!");
        p.write(param_file).expect("Unable to write parameters file!");
        p
    };

    info!("preparing verifying key");
    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);

    info!("Creating proofs...");
    let now = Instant::now();

    // Create an instance of circuit
    let c = dummy::DummyDemo::<Bls12> {
        xx: Fr::from_str("3")
    };

    // Create a groth16 proof with our parameters.
    let proof = create_random_proof(c, &params, rng).unwrap();
    info!("Total proof gen finished in {:04} seconds", now.elapsed().as_millis() as f64 / 1000.);

    info!("{}", verify_proof(
        &pvk,
        &proof,
        &[]
    ).unwrap());
}
