// The .goto is `kani fill.rs --only-codegen --keep-temps` of this file saved as fill.rs (Kani 0.68.0, CBMC 6.11.0).
// As cbmc_kani_fill_symbolic, but a digit before i keeps its 9. i < 17, as an
// empty tail never reaches the memset.
#[kani::proof]
fn harness() {
    let mut d: [u8; 17] = [b'9'; 17];
    let i: usize = kani::any();
    kani::assume(i < 17);
    unsafe { d.get_unchecked_mut(i..) }.fill(b'0');
    let j: usize = kani::any();
    kani::assume(j < 17);
    kani::assert(d[j] == b'0', "every digit is 0");
}
