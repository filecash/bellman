#[cfg(feature = "blst")]
use crate::bls::Engine;
#[cfg(feature = "blst")]
use blstrs::PairingCurveAffine;
use core::default::Default;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
#[cfg(feature = "pairing")]
use paired::{Engine, PairingCurveAffine};
use rand::rngs::OsRng;
use rayon::prelude::*;
/// PairingCheck represents a check of the form e(A,B)e(C,D)... = T. Checks can
/// be aggregated together using random linear combination. The efficiency comes
/// from keeping the results from the miller loop output before proceding to a final
/// exponentiation when verifying if all checks are verified.
/// It is a tuple:
/// - a miller loop result that is to be multiplied by other miller loop results
/// before going into a final exponentiation result
/// - a right side result which is already in the right subgroup Gt which is to
/// be compared to the left side when "final_exponentiatiat"-ed
pub struct PairingCheck<E: Engine>(E::Fqk, E::Fqk);

impl<E> PairingCheck<E>
where
    E: Engine,
{
    pub fn new() -> PairingCheck<E> {
        Self(E::Fqk::one(), E::Fqk::one())
    }

    pub fn new_invalid() -> PairingCheck<E> {
        Self(E::Fqk::one(), E::Fqk::zero())
    }

    /// Returns a pairing check from the output of the miller pairs and the expected
    /// right hand side such that the following must hold:
    /// $$
    /// \prod res = exp
    /// $$
    ///
    /// Note the check is NOT randomized and there must be only up to ONE check only that can not
    /// be randomized when merging.
    pub fn new_from_miller_output(result: E::Fqk, exp: E::Fqk) -> PairingCheck<E> {
        Self(result, exp)
    }

    /// returns a pairing tuple that is scaled by a random element.
    /// When aggregating pairing checks, this creates a random linear
    /// combination of all checks so that it is secure. Specifically
    /// we have e(A,B)e(C,D)... = out <=> e(g,h)^{ab + cd} = out
    /// We rescale using a random element $r$ to give
    /// e(rA,B)e(rC,D) ... = out^r <=>
    /// e(A,B)^r e(C,D)^r = out^r <=> e(g,h)^{abr + cdr} = out^r
    /// (e(g,h)^{ab + cd})^r = out^r
    pub fn new_random_from_miller_inputs<'a>(
        it: &[(&'a E::G1Affine, &'a E::G2Affine)],
        out: &'a E::Fqk,
    ) -> PairingCheck<E> {
        let coeff = derive_non_zero::<E>();
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
        PairingCheck(miller_out, outt)
    }

    /// takes another pairing tuple and combine both sides together. Note the checks are not
    /// randomized when merged, the checks must have been randomized before.
    pub fn merge(&mut self, p2: &PairingCheck<E>) {
        mul_if_not_one::<E>(&mut self.0, &p2.0);
        mul_if_not_one::<E>(&mut self.1, &p2.1);
    }

    pub fn verify(&self) -> bool {
        E::final_exponentiation(&self.0).unwrap() == self.1
    }
}

fn mul_if_not_one<E: Engine>(left: &mut E::Fqk, right: &E::Fqk) {
    let one = E::Fqk::one();
    if left == &one {
        *left = right.clone();
        return;
    } else if right == &one {
        // nothing to do here
        return;
    }
    left.mul_assign(right);
}

fn derive_non_zero<E: Engine>() -> E::Fr {
    let mut rng: OsRng = Default::default();
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
        let tuple = PairingCheck::<Bls12>::new_random_from_miller_inputs(
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
