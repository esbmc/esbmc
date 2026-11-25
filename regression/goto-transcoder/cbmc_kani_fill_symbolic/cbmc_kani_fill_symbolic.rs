// The .goto is `kani fill.rs --only-codegen --keep-temps` of this file saved as fill.rs (Kani 0.68.0, CBMC 6.11.0).
// The shape of core::num::flt2dec::round_up, `d[i + 1..].fill(b'0')`: fill
// the tail of a small digit buffer from a symbolic position.
#[kani::proof]
fn harness() {
    let mut d: [u8; 17] = [b'9'; 17];
    let i: usize = kani::any();
    kani::assume(i <= 17);
    unsafe { d.get_unchecked_mut(i..) }.fill(b'0');
    let j: usize = kani::any();
    kani::assume(j < 17);
    kani::assert(d[j] == if j < i { b'9' } else { b'0' }, "tail filled");
}
