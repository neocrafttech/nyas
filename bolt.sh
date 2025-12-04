#!/usr/bin/env bash
set -e

RUST_VERSION="1.91.0"

format() {
    cargo +nightly fmt;
}

setup_rust(){
    echo "[INFO] Checking Rust installation..."
    if command -v rustc >/dev/null 2>&1; then
        CURRENT_VERSION=$(rustc --version | awk '{print $2}')
        echo "[INFO] Found Rust version $CURRENT_VERSION"
        if [ "$CURRENT_VERSION" != "$RUST_VERSION" ]; then
            echo "[INFO] Updating Rust to $RUST_VERSION..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION
        else
            echo "[OK] Rust is already $RUST_VERSION"
        fi
    else
        echo "[INFO] Rust not found. Installing Rust $RUST_VERSION..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION
    fi

    export PATH="$HOME/.cargo/bin:$PATH"
    rustc --version
    cargo --version

    echo "[INFO] Installing cargo-nextest if missing..."
    if ! cargo nextest --version >/dev/null 2>&1; then
        cargo install cargo-nextest
    fi
    cargo nextest --version
}

setup() {
    setup_rust
}

check() {
    echo "[INFO] Running cargo check..."
    cargo check
    echo "[OK] Cargo check passed!"

    echo "[INFO] Checking code formatting..."
    cargo fmt -- --check
    echo "[OK] Code is properly formatted!"

    echo "[INFO] Running clippy lints..."
    cargo clippy -- -D warnings
    echo "[OK] Clippy checks passed!"
}

build() {
    echo "[INFO] Building..."
    cargo build --release
    echo "[OK] Build completed!"
}

test() {
    echo "[INFO] Testing workspace..."
    cargo nextest run
    echo "[OK] Testing completed!"
}

bench() {
    echo "[INFO] Running benchmarks..."
    cargo bench
    echo "[OK] Benchmarks completed!"
}

help() {
    echo "Usage: $0 [setup|check|build|test|bench|all|help]"
    echo
    echo "Commands:"
    echo "  setup   - Install Rust and cargo-nextest"
    echo "  format  - Format the code"
    echo "  check   - Run cargo check, fmt, and clippy"
    echo "  build   - Only build the workspace (runs check first)"
    echo "  test    - Only run tests"
    echo "  bench   - Only run benchmarks"
    echo "  all     - Run check, build, and test"
    echo "  help    - Show this help message"
}

main() {
    cmd="$1"
    case "$cmd" in
        setup)
            setup
            ;;
        format)
            format
            ;;
        check)
            check
            ;;
        build)
            build
            ;;
        test)
            test
            ;;
        bench)
            bench
            ;;
        all)
            setup
            check
            build
            deploy
            ;;
        help|""|*)
            help
            ;;
    esac
}

main "$@"
