use crate::vector_data::VectorData;
use crate::{vector_store::VectorStore, writer::Writer};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;

pub type Id = usize;

/// Trait for types that can be used as database key suffixes
pub trait KeySuffix: Send + Sync + Sized {
    fn encode(&self, buf: &mut Vec<u8>);
    fn decode(raw: &[u8]) -> Result<Self, DecodeError>;

    /// Estimate encoded size for pre-allocation
    fn encoded_size_hint(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum DecodeError {
    #[error("Invalid buffer length: expected {expected}, got {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("Invalid UTF-8: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("Decode error: {0}")]
    Other(String),
}

///Use big-endian to encode and decode for lexical ordering
impl KeySuffix for Id {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_be_bytes());
    }

    #[inline]
    fn decode(raw: &[u8]) -> Result<Self, DecodeError> {
        raw.try_into().map(Self::from_be_bytes).map_err(|_| DecodeError::InvalidLength {
            expected: mem::size_of::<Id>(),
            actual: raw.len(),
        })
    }

    #[inline]
    fn encoded_size_hint(&self) -> usize {
        mem::size_of::<Id>()
    }
}

///Use big-endian to encode and decode for lexical ordering
impl KeySuffix for u64 {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_be_bytes());
    }

    #[inline]
    fn decode(raw: &[u8]) -> Result<Self, DecodeError> {
        raw.try_into().map(Self::from_be_bytes).map_err(|_| DecodeError::InvalidLength {
            expected: mem::size_of::<u64>(),
            actual: raw.len(),
        })
    }

    #[inline]
    fn encoded_size_hint(&self) -> usize {
        mem::size_of::<u64>()
    }
}

///String bytes are in lexical ordering
impl KeySuffix for String {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(self.as_bytes());
    }

    #[inline]
    fn decode(raw: &[u8]) -> Result<Self, DecodeError> {
        String::from_utf8(raw.to_vec()).map_err(DecodeError::from)
    }

    #[inline]
    fn encoded_size_hint(&self) -> usize {
        self.len()
    }
}

///Vec<u8> bytes are in lexical ordering
impl KeySuffix for Vec<u8> {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(self);
    }

    #[inline]
    fn decode(raw: &[u8]) -> Result<Self, DecodeError> {
        Ok(raw.to_vec())
    }

    #[inline]
    fn encoded_size_hint(&self) -> usize {
        self.len()
    }
}

impl KeySuffix for () {
    #[inline]
    fn encode(&self, _: &mut Vec<u8>) {}

    #[inline]
    fn decode(_: &[u8]) -> Result<Self, DecodeError> {
        Ok(())
    }
}

/// Represents a typed database entity with prefix-based keys
pub struct Entity<K: KeySuffix, V: DeserializeOwned + Send + Serialize + Sync> {
    prefix: u8,
    key: PhantomData<fn() -> K>,
    val: PhantomData<fn() -> V>,
}

impl<K: KeySuffix, V: DeserializeOwned + Send + Serialize + Sync> Entity<K, V> {
    /// Create a new entity type with the given prefix
    pub const fn new(prefix: u8) -> Self {
        Self { prefix, key: PhantomData, val: PhantomData }
    }

    /// Get the prefix for this entity type
    #[inline]
    pub const fn prefix(&self) -> u8 {
        self.prefix
    }

    /// Encode a key with the entity prefix
    #[inline]
    pub fn encode_key(&self, key: impl Borrow<K>) -> Vec<u8> {
        let key_ref = key.borrow();
        let mut buf = Vec::with_capacity(1 + key_ref.encoded_size_hint());
        buf.push(self.prefix);
        key_ref.encode(&mut buf);
        buf
    }

    /// Decode a key suffix (without prefix)
    #[inline]
    pub fn decode_key(&self, raw: &[u8]) -> Result<K, DecodeError> {
        K::decode(raw)
    }

    /// Encode a value using MessagePack
    #[inline]
    pub fn encode_value(&self, value: impl Borrow<V>) -> Vec<u8> {
        rmp_serde::to_vec_named(value.borrow()).expect("serialization failed")
    }

    /// Decode a value from MessagePack
    #[inline]
    pub fn decode_value(&self, raw: &[u8]) -> Result<V, rmp_serde::decode::Error> {
        rmp_serde::from_slice(raw)
    }

    /// Read a single entity by key
    pub fn get(&self, vector_store: &dyn VectorStore, key: impl Borrow<K>) -> Option<V> {
        let raw_key = self.encode_key(key);
        vector_store.get(&raw_key).and_then(|raw_val| self.decode_value(&raw_val).ok())
    }

    /// Read multiple entities by keys
    pub fn multi_get<'a, KeyItem, Keys>(
        &self, vector_store: &dyn VectorStore, keys: Keys,
    ) -> Vec<Option<V>>
    where
        KeyItem: Borrow<K> + 'a,
        Keys: IntoIterator<Item = KeyItem>,
    {
        let raw_keys: Vec<_> = keys.into_iter().map(|k| self.encode_key(k)).collect();
        vector_store
            .multi_get(&raw_keys)
            .into_iter()
            .map(|opt_raw| opt_raw.and_then(|raw| self.decode_value(&raw).ok()))
            .collect()
    }

    /// Check if a key exists
    #[inline]
    pub fn exists(&self, vector_store: &dyn VectorStore, key: impl Borrow<K>) -> bool {
        let raw_key = self.encode_key(key);
        vector_store.get(&raw_key).is_some()
    }

    /// Iterate over all entities with this prefix
    pub fn iter<'a>(
        &'a self, vector_store: &'a dyn VectorStore,
    ) -> impl Iterator<Item = (K, V)> + Send + Sync + 'a {
        vector_store.iter(self.prefix).filter_map(move |(raw_key, raw_val)| {
            // Skip the prefix byte
            let key = self.decode_key(&raw_key[1..]).ok()?;
            let val = self.decode_value(&raw_val).ok()?;
            Some((key, val))
        })
    }

    /// Write a single entity
    #[inline]
    pub fn put(&self, vector_store: &dyn VectorStore, key: impl Borrow<K>, value: impl Borrow<V>) {
        let raw_key = self.encode_key(key);
        let raw_val = self.encode_value(value);
        vector_store.put(raw_key, raw_val);
    }

    /// Delete a single entity
    #[inline]
    pub fn delete(&self, vector_store: &dyn VectorStore, key: impl Borrow<K>) {
        let raw_key = self.encode_key(key);
        vector_store.delete(raw_key);
    }

    /// Add a put operation to a batch
    #[inline]
    pub fn batch_put(&self, batch: &mut Vec<Writer>, key: impl Borrow<K>, value: impl Borrow<V>) {
        let raw_key = self.encode_key(key);
        let raw_val = self.encode_value(value);
        batch.push(Writer::put(raw_key, raw_val));
    }

    /// Add a delete operation to a batch
    #[inline]
    pub fn batch_delete(&self, batch: &mut Vec<Writer>, key: impl Borrow<K>) {
        let raw_key = self.encode_key(key);
        batch.push(Writer::delete(raw_key));
    }

    /// Write multiple entities in a batch
    pub fn batch_put_many<'a, Item, Items>(&self, batch: &mut Vec<Writer>, items: Items)
    where
        Item: Borrow<(K, V)> + 'a,
        Items: IntoIterator<Item = Item>,
    {
        for item in items {
            let (key, val) = item.borrow();
            self.batch_put(batch, key, val);
        }
    }

    /// Delete multiple entities in a batch
    pub fn batch_delete_many<'a, KeyItem, Keys>(&self, batch: &mut Vec<Writer>, keys: Keys)
    where
        KeyItem: Borrow<K> + 'a,
        Keys: IntoIterator<Item = KeyItem>,
    {
        for key in keys {
            self.batch_delete(batch, key);
        }
    }
}

// Make Entity Debug if the key and value types support it
impl<K: KeySuffix, V: DeserializeOwned + Send + Serialize + Sync> std::fmt::Debug for Entity<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entity")
            .field("prefix", &self.prefix)
            .field("key_type", &std::any::type_name::<K>())
            .field("value_type", &std::any::type_name::<V>())
            .finish()
    }
}

/// Node data stored in the database
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeData {
    /// Version counter for optimistic concurrency control
    pub version: u64,
    /// IDs of neighboring nodes
    pub neighbors: Vec<Id>,
    /// Vector data (shared via Arc for efficient cloning)
    pub vector: Arc<VectorData>,
}

impl NodeData {
    /// Create new node data
    pub fn new(version: u64, neighbors: Vec<Id>, vector: VectorData) -> Self {
        Self { version, neighbors, vector: Arc::new(vector) }
    }

    /// Increment version and return the new version
    #[inline]
    pub fn increment_version(&mut self) -> u64 {
        self.version += 1;
        self.version
    }

    /// Try to unwrap the vector if this is the only reference
    pub fn try_into_vector(self) -> Result<VectorData, Self> {
        Arc::try_unwrap(self.vector).map_err(|arc| Self {
            version: self.version,
            neighbors: self.neighbors,
            vector: arc,
        })
    }

    /// Clone the vector data
    #[inline]
    pub fn clone_vector(&self) -> VectorData {
        (*self.vector).clone()
    }

    /// Get a reference to the vector
    #[inline]
    pub fn vector(&self) -> &VectorData {
        &self.vector
    }

    /// Get the number of neighbors
    #[inline]
    pub fn neighbor_count(&self) -> usize {
        self.neighbors.len()
    }
}

/// Create a write batch from multiple operations
pub fn create_batch<F>(f: F) -> Vec<Writer>
where
    F: FnOnce(&mut Vec<Writer>),
{
    let mut batch = Vec::new();
    f(&mut batch);
    batch
}

/// Execute a batch write operation
#[inline]
pub fn execute_batch(vector_store: &dyn VectorStore, batch: Vec<Writer>) {
    if !batch.is_empty() {
        vector_store.write(&batch);
    }
}
