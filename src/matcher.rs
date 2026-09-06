//! Allocation-free rejection filters. Full encoding is only needed after a hit.
const ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

pub struct EthMatcher {
    checks: Vec<(usize, u8, u8)>,
}

impl EthMatcher {
    pub fn new(prefix: &str, suffix: &str) -> Self {
        let checks = [(0, prefix), (40 - suffix.len(), suffix)]
            .into_iter()
            .flat_map(|(offset, pattern)| {
                pattern.bytes().enumerate().map(move |(i, c)| {
                    let n = (c as char).to_digit(16).expect("validated hex") as u8;
                    let pos = offset + i;
                    let shift = if pos % 2 == 0 { 4 } else { 0 };
                    (pos / 2, 15 << shift, n << shift)
                })
            })
            .collect();
        Self { checks }
    }

    pub fn matches(&self, address: &[u8]) -> bool {
        self.checks
            .iter()
            .all(|&(i, mask, value)| address[i] & mask == value)
    }
}

pub struct SolMatcher<'a> {
    prefix: &'a [u8],
    suffix: &'a [u8],
    ignore_case: bool,
}

impl<'a> SolMatcher<'a> {
    pub fn new(prefix: &'a str, suffix: &'a str, ignore_case: bool) -> Self {
        Self {
            prefix: prefix.as_bytes(),
            suffix: suffix.as_bytes(),
            ignore_case,
        }
    }

    fn equal(&self, a: &[u8], b: &[u8]) -> bool {
        if self.ignore_case {
            a.eq_ignore_ascii_case(b)
        } else {
            a == b
        }
    }

    #[cfg(test)]
    pub fn matches(&self, key: &[u8; 32], output: &mut [u8; 44]) -> Option<usize> {
        self.matches_counted(key, output, &mut 0)
    }

    pub fn matches_counted(
        &self,
        key: &[u8; 32],
        output: &mut [u8; 44],
        encodes: &mut u64,
    ) -> Option<usize> {
        if !self.suffix.is_empty() {
            // 58^4 fits in 24 bits. Processing four bytes at a time stays
            // below 2^56, avoiding overflow and expensive u128 division.
            const MOD: u64 = 58 * 58 * 58 * 58;
            let mut rem = 0u64;
            for chunk in key.chunks_exact(4) {
                rem = ((rem << 32) | u32::from_be_bytes(chunk.try_into().unwrap()) as u64) % MOD;
            }
            for &c in self.suffix.iter().rev().take(4) {
                let actual = ALPHABET[(rem % 58) as usize];
                if !self.equal(&[actual], &[c]) {
                    return None;
                }
                rem /= 58;
            }
        }
        *encodes += 1;
        let len = bs58::encode(key)
            .onto(&mut output[..])
            .expect("32 bytes fit in 44 Base58 digits");
        if len < self.prefix.len() || len < self.suffix.len() {
            return None;
        }
        (self.equal(&output[..self.prefix.len()], self.prefix)
            && self.equal(&output[len - self.suffix.len()..len], self.suffix))
        .then_some(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, SeedableRng};

    #[test]
    fn differential_matchers() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(19);
        for sample in 0..256 {
            let mut key = [0; 32];
            rng.fill_bytes(&mut key);
            if sample < 33 {
                key[..sample].fill(0);
            }
            let hex = hex::encode(&key[..20]);
            let base58 = bs58::encode(key).into_string();
            for len in 0..=40 {
                let prefix = &hex[..len];
                let suffix = &hex[40 - len..];
                assert!(EthMatcher::new(prefix, suffix).matches(&key[..20]));
                assert_eq!(
                    EthMatcher::new("abc", suffix).matches(&key[..20]),
                    hex.starts_with("abc")
                );
            }
            for len in 0..=base58.len() {
                for ignore in [false, true] {
                    let prefix = &base58[..len];
                    let suffix = &base58[base58.len() - len..];
                    let mut out = [0; 44];
                    assert!(SolMatcher::new(prefix, suffix, ignore)
                        .matches(&key, &mut out)
                        .is_some());
                    assert_eq!(
                        SolMatcher::new("", "AbCd", ignore)
                            .matches(&key, &mut out)
                            .is_some(),
                        if ignore {
                            base58.to_ascii_lowercase().ends_with("abcd")
                        } else {
                            base58.ends_with("AbCd")
                        }
                    );
                    assert!(SolMatcher::new(
                        &prefix.to_ascii_lowercase(),
                        &suffix.to_ascii_lowercase(),
                        true
                    )
                    .matches(&key, &mut out)
                    .is_some());
                }
            }
        }
    }
}
