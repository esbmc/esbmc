// The .goto is `kani bounded.rs --only-codegen --keep-temps` of this file saved as bounded.rs (Kani 0.68.0 at 0aac16e, CBMC 6.11.0).
// i == 5 needs six loop iterations.
#[kani::proof]
fn harness() {
    let n: u8 = kani::any();
    let mut i: u8 = 0;
    while i < n {
        i += 1;
    }
    kani::cover!(i == 5);
}
