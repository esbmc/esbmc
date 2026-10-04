// The .goto is `kani pass.rs --only-codegen --keep-temps` of this file saved as pass.rs (Kani 0.68.0, CBMC 6.11.0).
// Kani tags the assertion with a reachability_check and kani::cover with cover.
#[kani::proof]
fn harness() {
    let x: u8 = kani::any();
    kani::cover!(x > 5);
    assert!((x as u16) + 1 > x as u16);
}
