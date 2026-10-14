// The .goto is `kani popfail.rs --only-codegen --keep-temps` of this file saved as popfail.rs (Kani 0.68.0, CBMC 6.11.0).
// Kani lowers ctpop to a u32-typed popcount, assigned here where the branches
// merge.
#![feature(core_intrinsics)]
#![allow(internal_features)]
#[kani::proof]
fn harness() {
    let s: u64 = 8;
    let mut x: u32 = 0;
    if kani::any() {
        x = core::intrinsics::ctpop(!s & (s - 1));
    }
    assert!(x <= 2);
}
