#[cfg(feature = "blst")]
use crate::bls::Engine;
#[cfg(feature = "blst")]
use blstrs::PairingCurveAffine;

use crossbeam_channel::{bounded, Receiver, Sender};
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
#[cfg(feature = "pairing")]
use paired::{Engine, PairingCurveAffine};
use rayon::prelude::*;

use std::sync::{
    atomic::{AtomicBool, Ordering::SeqCst},
    Arc, Mutex,
};

#[derive(Debug)]
pub struct PairingChecks<E: Engine, R: rand::RngCore + Send> {
    /// Circuit breaker to allow canceling all checks and marking the whole check as failed.
    valid: Arc<AtomicBool>,
    merge_send: Sender<PairingCheck<E>>,
    valid_recv: Receiver<bool>,
    /// Random number generator used for generating the random coefficients.
    rng: Mutex<R>,
    /// Ensures that the non randomized check is only added exactly once.
    non_random_check_done: AtomicBool,
}

impl<E: Engine, R: rand::RngCore + Send> PairingChecks<E, R> {
    pub fn new(rng: R) -> Self {
        let (merge_send, merge_recv): (Sender<PairingCheck<E>>, Receiver<PairingCheck<E>>) =
            bounded(10);
        let (valid_send, valid_recv) = bounded(1);

        let valid = Arc::new(AtomicBool::new(true));
        let valid_copy = valid.clone();

        rayon::spawn(move || {
            let mut acc = PairingCheck::new();
            while let Ok(tuple) = merge_recv.recv() {
                // only do work as long as we know we are still valid
                if valid_copy.load(SeqCst) {
                    acc.merge(&tuple);
                }
            }
            if valid_copy.load(SeqCst) {
                valid_send.send(acc.verify()).expect("failed to send");
            }
        });

        PairingChecks {
            valid,
            merge_send,
            valid_recv,
            rng: Mutex::new(rng),
            non_random_check_done: AtomicBool::new(false),
        }
    }

    /// Fails the whole check.
    pub fn invalidate(&self) {
        self.valid.store(false, SeqCst);
    }

    fn merge_pair(&self, result: E::Fqk, exp: E::Fqk, must_randomize: bool) {
        self.merge(PairingCheck::from_pair(result, exp), must_randomize);
    }

    fn merge_miller_one(&self, result: E::Fqk, must_randomize: bool) {
        self.merge(PairingCheck::from_miller_one(result), must_randomize);
    }

    pub fn merge_pairing_equation(&self, left: Vec<E::Fqk>, pair: (E::Fqk, E::Fqk)) {
        let must_randomize = self.non_random_check_done.load(SeqCst);

        for l in left.iter() {
            self.merge_miller_one(*l, must_randomize);
        }
        self.merge_pair(pair.0, pair.1, must_randomize);
    }

    pub fn merge_miller_inputs<'a>(
        &self,
        it: &[(&'a E::G1Affine, &'a E::G2Affine)],
        out: &'a E::Fqk,
    ) {
        let must_randomize = self.non_random_check_done.load(SeqCst);
        let coeff = {
            let rng: &mut R = &mut self.rng.lock().unwrap();
            derive_non_zero::<E, _>(rng)
        };
        self.merge(
            PairingCheck::from_miller_inputs(coeff, it, out),
            must_randomize,
        );
    }

    fn merge(&self, check: PairingCheck<E>, must_randomize: bool) {
        if !check.randomized {
            assert!(
                !must_randomize,
                "Cannot merge non-randomized check with must_randomize true."
            );
            self.non_random_check_done.store(true, SeqCst);
        };

        self.merge_send.send(check).unwrap();
    }

    pub fn verify(self) -> bool {
        let Self {
            valid,
            merge_send,
            valid_recv,
            ..
        } = self;

        drop(merge_send); // stop the merge process

        valid.load(SeqCst) && valid_recv.recv().unwrap_or_default()
    }
}

/// PairingCheck represents a check of the form e(A,B)e(C,D)... = T. Checks can
/// be aggregated together using random linear combination. The efficiency comes
/// from keeping the results from the miller loop output before proceding to a final
/// exponentiation when verifying if all checks are verified.
/// It is a tuple:
/// - a miller loop result that is to be multiplied by other miller loop results
/// before going into a final exponentiation result
/// - a right side result which is already in the right subgroup Gt which is to
/// be compared to the left side when "final_exponentiatiat"-ed
#[derive(Debug)]
struct PairingCheck<E: Engine> {
    left: E::Fqk,
    right: E::Fqk,
    randomized: bool,
}

impl<E> PairingCheck<E>
where
    E: Engine,
{
    fn new() -> PairingCheck<E> {
        Self {
            left: E::Fqk::one(),
            right: E::Fqk::one(),
            randomized: false,
        }
    }

    fn from_pair(result: E::Fqk, exp: E::Fqk) -> PairingCheck<E> {
        Self {
            left: result,
            right: exp,
            randomized: false,
        }
    }

    fn from_miller_one(result: E::Fqk) -> PairingCheck<E> {
        Self {
            left: result,
            right: E::Fqk::one(),
            randomized: false,
        }
    }

    /// returns a pairing tuple that is scaled by a random element.
    /// When aggregating pairing checks, this creates a random linear
    /// combination of all checks so that it is secure. Specifically
    /// we have e(A,B)e(C,D)... = out <=> e(g,h)^{ab + cd} = out
    /// We rescale using a random element $r$ to give
    /// e(rA,B)e(rC,D) ... = out^r <=>
    /// e(A,B)^r e(C,D)^r = out^r <=> e(g,h)^{abr + cdr} = out^r
    /// (e(g,h)^{ab + cd})^r = out^r
    fn from_miller_inputs<'a>(
        coeff: E::Fr,
        it: &[(&'a E::G1Affine, &'a E::G2Affine)],
        out: &'a E::Fqk,
    ) -> PairingCheck<E> {
        let miller_out = it
            .into_par_iter()
            .map(|(a, b)| {
                let na = a.mul(coeff).into_affine();
                (na.prepare(), b.prepare())
            })
            .map(|(a, b)| E::miller_loop(&[(&a, &b)]))
            .fold(
                || E::Fqk::one(),
                |mut acc, res| {
                    acc.mul_assign(&res);
                    acc
                },
            )
            .reduce(
                || E::Fqk::one(),
                |mut acc, res| {
                    acc.mul_assign(&res);
                    acc
                },
            );
        let mut outt = out.clone();
        if out != &E::Fqk::one() {
            // we only need to make this expensive operation is the output is
            // not one since 1^r = 1
            outt = outt.pow(&coeff.into_repr());
        }
        PairingCheck {
            left: miller_out,
            right: outt,
            randomized: true,
        }
    }

    /// takes another pairing tuple and combine both sides together as a random
    /// linear combination.
    fn merge(&mut self, p2: &PairingCheck<E>) {
        // multiply miller loop results together
        self.left.mul_assign(&p2.left);
        // multiply right side in GT together
        if p2.right != E::Fqk::one() {
            if self.right != E::Fqk::one() {
                // if both sides are not one, then multiply
                self.right.mul_assign(&p2.right);
            } else {
                // otherwise, only keep the side which is not one
                self.right = p2.right.clone();
            }
        }
        // A merged PairingCheck is only randomized if both of its contributors are.
        self.randomized = self.randomized && p2.randomized;
        // if p2.1 is one, then we don't need to change anything.
    }

    fn verify(&self) -> bool {
        E::final_exponentiation(&self.left).unwrap() == self.right
    }
}

fn derive_non_zero<E: Engine, R: rand::RngCore>(mut rng: R) -> E::Fr {
    loop {
        let coeff = E::Fr::random(&mut rng);
        if coeff != E::Fr::zero() {
            return coeff;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[cfg(feature = "blst")]
    use crate::bls::{Bls12, G1Projective, G2Projective};
    use groupy::CurveProjective;
    use rand_core::RngCore;
    use rand_core::SeedableRng;

    #[cfg(feature = "pairing")]
    use paired::bls12_381::{Bls12, G1 as G1Projective, G2 as G2Projective};

    fn gen_pairing_check<R: RngCore>(r: &mut R) -> PairingCheck<Bls12> {
        let g1r = G1Projective::random(r);
        let g2r = G2Projective::random(r);
        let exp = Bls12::pairing(g1r.clone(), g2r.clone());
        let coeff = derive_non_zero::<Bls12, _>(r);
        let tuple = PairingCheck::<Bls12>::from_miller_inputs(
            coeff,
            &[(&g1r.into_affine(), &g2r.into_affine())],
            &exp,
        );
        assert!(tuple.verify());
        tuple
    }
    #[test]
    fn test_pairing_randomize() {
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let tuples = (0..3)
            .map(|_| gen_pairing_check(&mut rng))
            .collect::<Vec<_>>();
        let final_tuple = tuples
            .iter()
            .fold(PairingCheck::<Bls12>::new(), |mut acc, tu| {
                acc.merge(&tu);
                acc
            });
        assert!(final_tuple.verify());
    }
}
