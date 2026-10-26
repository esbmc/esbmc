// The .goto is `kani arr_too_large.rs --only-codegen --keep-temps` of this file saved as arr_too_large.rs (Kani 0.68.0, CBMC 6.11.0).
// A 2^21-element array read element by element is over the bound.
#[kani::proof]
fn harness() {
    let big = [0u16; 1 << 20];
    let b: [u8; 1 << 21] = unsafe { *(&big as *const [u16; 1 << 20] as *const [u8; 1 << 21]) };
    assert!(b[0] == 0);
}
