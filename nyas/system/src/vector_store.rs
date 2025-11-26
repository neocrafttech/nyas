use std::borrow::Borrow;
use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::entity::{Entity, KeySuffix};
use crate::writer::{BatchWriter, Writer};

pub type VectorIter<'a> = Box<dyn Iterator<Item = (Box<[u8]>, Box<[u8]>)> + Send + Sync + 'a>;

pub trait VectorStore: Send + Sync {
    fn get(&self, key: &[u8]) -> Option<Vec<u8>>;
    fn put(&self, key: Vec<u8>, value: Vec<u8>);
    fn delete(&self, key: Vec<u8>);
    fn iter(&self, prefix: u8) -> VectorIter<'_>;
    fn multi_get(&self, keys: &[Vec<u8>]) -> Vec<Option<Vec<u8>>>;
    fn write(&self, writers: &[Writer]);

    fn exists(&self, key: &[u8]) -> bool {
        self.get(key).is_some()
    }

    fn write_batch(&self, batch: BatchWriter) {
        self.write(batch.writers());
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum VectorStoreError {
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Deserialization error: {0}")]
    Deserialization(String),
    #[error("Key not found")]
    KeyNotFound,
    #[error("Storage error: {0}")]
    Storage(String),
}

pub trait VectorStoreExt: VectorStore {
    /// Read multiple entities by keys
    fn read_entities<K, V, KeysItem, Keys>(
        &self, ent: &'static Entity<K, V>, keys: Keys,
    ) -> Vec<Option<V>>
    where
        K: KeySuffix,
        V: DeserializeOwned + Send + Serialize + Sync,
        KeysItem: Borrow<K>,
        Keys: IntoIterator<Item = KeysItem>,
    {
        let key_vec: Vec<_> = keys.into_iter().map(|k| ent.encode_key(k)).collect();
        self.multi_get(&key_vec)
            .into_iter()
            .map(|raw| raw.and_then(|r| ent.decode_value(&r).ok()))
            .collect()
    }

    /// Read a single entity by key
    fn read_entity<K, V>(&self, entity: &'static Entity<K, V>, key: &K) -> Option<V>
    where
        K: KeySuffix,
        V: DeserializeOwned + Send + Serialize + Sync,
    {
        self.get(&entity.encode_key(key)).and_then(|raw| entity.decode_value(&raw).ok())
    }
}

// Blanket implementation for all Store types
impl<T: VectorStore + ?Sized> VectorStoreExt for T {}

/// Arc-wrapped store for cheap cloning
pub type SharedStore = Arc<dyn VectorStore>;

/// Helper to create a shared store
pub fn shared<S: VectorStore + 'static>(store: S) -> SharedStore {
    Arc::new(store)
}
