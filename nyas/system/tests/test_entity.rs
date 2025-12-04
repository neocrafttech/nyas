use std::sync::Arc;

use system::entity::{Entity, Id, KeySuffix, NodeData};
use system::vector_data::VectorData;

#[test]
fn test_key_encoding() {
    let id: Id = 42;
    let mut buf = Vec::new();
    id.encode(&mut buf);
    assert_eq!(buf.len(), 8);
    assert_eq!(Id::decode(&buf).unwrap(), id);
}

#[test]
fn test_string_key() {
    let key = "test_key".to_string();
    let mut buf = Vec::new();
    key.encode(&mut buf);
    assert_eq!(String::decode(&buf).unwrap(), key);
}

#[test]
fn test_entity_key_encoding() {
    let entity = Entity::<Id, String>::new(10);
    let key: Id = 123;
    let encoded = entity.encode_key(&key);

    assert_eq!(encoded[0], 10); // prefix
    assert_eq!(encoded.len(), 9); // prefix + 8 bytes for u64
    assert_eq!(entity.decode_key(&encoded[1..]).unwrap(), key);
}

#[test]
fn test_node_data() {
    let mut node =
        NodeData { version: 1, neighbors: vec![1, 2, 3], vector: Arc::new(VectorData::default()) };

    assert_eq!(node.version, 1);
    assert_eq!(node.increment_version(), 2);
    assert_eq!(node.version, 2);
    assert_eq!(node.neighbor_count(), 3);
}
