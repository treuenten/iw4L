use std::hash::{Hash, Hasher};

use bevy::platform::collections::HashSet;
use bevy::prelude::Resource;

pub(crate) const DYNAMIC_LIGHT_TECHS: [u8; 3] = [
    lighting_iw4::TECHNIQUE_LIGHT_SPOT,
    lighting_iw4::TECHNIQUE_LIGHT_OMNI,
    lighting_iw4::TECHNIQUE_LIGHT_SPOT_SHADOW,
];

pub(crate) const PREWARM_ENV: &str = "IW4L_PREWARM";

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| perf::switch(PREWARM_ENV))
}

const LEARNED_KIND: &str = "prewarm";
// Bump the version whenever the pipeline key changes shape or meaning.
const LEARNED_KEY: &str = "exact-first-use-v1";
const LEARNED_CAP: usize = 4096;

// Saved across runs, so this must give the same number for the same key in
// every run; a randomly seeded hasher would not.
pub(crate) fn key_hash(key: &impl Hash) -> u64 {
    struct Fnv(u64);
    impl Hasher for Fnv {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, bytes: &[u8]) {
            for byte in bytes {
                self.0 ^= u64::from(*byte);
                self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    let mut hasher = Fnv(0xcbf2_9ce4_8422_2325);
    key.hash(&mut hasher);
    hasher.finish()
}

#[derive(Resource, Default)]
pub(crate) struct LearnedFirstUse {
    loaded: bool,
    order: Vec<u64>,
    set: HashSet<u64>,
}

impl LearnedFirstUse {
    fn load(&mut self) {
        if std::mem::replace(&mut self.loaded, true) {
            return;
        }
        let Some(bytes) = asset_material::cache_get(LEARNED_KIND, LEARNED_KEY) else {
            return;
        };
        self.absorb(&parse(&bytes));
    }

    fn absorb(&mut self, hashes: &[u64]) -> bool {
        let mut added = false;
        for &hash in hashes {
            if self.set.insert(hash) {
                self.order.push(hash);
                added = true;
            }
        }
        if self.order.len() > LEARNED_CAP {
            let drop = self.order.len() - LEARNED_CAP;
            for hash in self.order.drain(..drop) {
                self.set.remove(&hash);
            }
        }
        added
    }

    pub(crate) fn contains(&mut self, hash: u64) -> bool {
        self.load();
        self.set.contains(&hash)
    }

    pub(crate) fn len(&mut self) -> usize {
        self.load();
        self.order.len()
    }

    pub(crate) fn remember(&mut self, hashes: &[u64]) {
        self.load();
        if !self.absorb(hashes) {
            return;
        }
        if let Err(error) =
            asset_material::cache_put(LEARNED_KIND, LEARNED_KEY, &format(&self.order))
        {
            diag::warn!(
                World,
                "prewarm: could not save the first-use pipeline list: {error}"
            );
        }
    }
}

fn parse(bytes: &[u8]) -> Vec<u64> {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter_map(|line| u64::from_str_radix(line.trim(), 16).ok())
        .collect()
}

fn format(hashes: &[u64]) -> Vec<u8> {
    let mut out = String::with_capacity(hashes.len() * 17);
    for hash in hashes {
        out.push_str(&format!("{hash:016x}\n"));
    }
    out.into_bytes()
}
