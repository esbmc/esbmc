// The .goto is `kani fail.rs --only-codegen --keep-temps` of this file saved as fail.rs (Kani 0.68.0, CBMC 6.11.0).
// The same shape with an assertion that fails for x == 255.
#[kani::proof]
fn harness() {
    let x: u8 = kani::any();
    kani::cover!(x > 5);
    assert!(x < 255);
}
