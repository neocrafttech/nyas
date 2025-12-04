#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Writer {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}

impl Writer {
    #[inline]
    pub fn put(key: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>) -> Self {
        Self::Put { key: key.into(), value: value.into() }
    }

    #[inline]
    pub fn delete(key: impl Into<Vec<u8>>) -> Self {
        Self::Delete { key: key.into() }
    }

    #[inline]
    pub fn key(&self) -> &[u8] {
        match self {
            Self::Put { key, .. } | Self::Delete { key } => key,
        }
    }

    #[inline]
    pub fn is_put(&self) -> bool {
        matches!(self, Self::Put { .. })
    }

    #[inline]
    pub fn is_delete(&self) -> bool {
        matches!(self, Self::Delete { .. })
    }
}

#[derive(Default, Debug)]
pub struct BatchWriter {
    writers: Vec<Writer>,
}

impl BatchWriter {
    pub fn new() -> Self {
        Self { writers: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { writers: Vec::with_capacity(capacity) }
    }

    pub fn put(&mut self, key: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>) {
        self.writers.push(Writer::put(key, value));
    }

    pub fn delete(&mut self, key: impl Into<Vec<u8>>) {
        self.writers.push(Writer::delete(key));
    }

    pub fn len(&self) -> usize {
        self.writers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.writers.is_empty()
    }

    pub fn clear(&mut self) {
        self.writers.clear();
    }

    pub fn writers(&self) -> &Vec<Writer> {
        &self.writers
    }
}
