doc:
    rm -rf docs
    cargo doc --no-deps --target-dir docs/rust-tmp
    mkdir -p docs/rust
    mv docs/rust-tmp/doc/* docs/rust/
    rm -rf docs/rust-tmp
