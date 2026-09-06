use secp256k1::{PublicKey, Scalar, Secp256k1, SecretKey, Signing};

/// One independent 256-bit random starting key per worker and search round.
/// Discard the walk after a hit: never export two related keys from one walk.
pub struct EthWalk {
    secret: SecretKey,
    public: PublicKey,
    generator: PublicKey,
}

impl EthWalk {
    pub fn new<C: Signing>(secp: &Secp256k1<C>, secret: SecretKey) -> Self {
        Self {
            secret,
            public: PublicKey::from_secret_key(secp, &secret),
            generator: PublicKey::from_secret_key(
                secp,
                &SecretKey::from_slice(&Scalar::ONE.to_be_bytes()).unwrap(),
            ),
        }
    }

    pub fn current(&self) -> ([u8; 32], [u8; 65]) {
        (
            self.secret.secret_bytes(),
            self.public.serialize_uncompressed(),
        )
    }

    pub fn advance(&mut self) {
        if let Ok(next) = self.secret.add_tweak(&Scalar::ONE) {
            self.secret = next;
            self.public = self
                .public
                .combine(&self.generator)
                .expect("nonzero next scalar");
        } else {
            // n - 1 + 1 is the point at infinity. Skip invalid scalar zero.
            self.secret = SecretKey::from_slice(&Scalar::ONE.to_be_bytes()).unwrap();
            self.public = self.generator;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn walk_matches_independent_scalar_multiplication_and_wraps() {
        let secp = Secp256k1::new();
        for start in [
            [42; 32],
            hex::decode("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140")
                .unwrap()
                .try_into()
                .unwrap(),
        ] {
            let mut walk = EthWalk::new(&secp, SecretKey::from_slice(&start).unwrap());
            for _ in 0..1024 {
                let (sk, pk) = walk.current();
                assert_eq!(
                    pk,
                    PublicKey::from_secret_key(&secp, &SecretKey::from_slice(&sk).unwrap())
                        .serialize_uncompressed()
                );
                walk.advance();
            }
        }
    }
}
