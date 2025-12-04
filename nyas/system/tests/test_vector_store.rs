use system::writer::{BatchWriter, Writer};

#[test]
fn test_batch_writer() {
    let mut batch = BatchWriter::new();
    batch.put(b"key1".to_vec(), b"value1".to_vec());
    batch.put(b"key2".to_vec(), b"value2".to_vec());
    batch.delete(b"key3".to_vec());

    assert_eq!(batch.len(), 3);
    assert!(!batch.is_empty());
}

#[test]
fn test_writer() {
    let put_writer = Writer::put(b"key".to_vec(), b"value".to_vec());
    assert!(put_writer.is_put());
    assert!(!put_writer.is_delete());
    assert_eq!(put_writer.key(), b"key");

    let del_op = Writer::delete(b"key".to_vec());
    assert!(del_op.is_delete());
    assert!(!del_op.is_put());
}
