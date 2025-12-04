use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    tonic_prost_build::configure()
        .file_descriptor_set_path(out_dir.join("vector_descriptor.bin"))
        .compile_protos(&["vector.proto"], &["../proto"])
        .unwrap();
}
