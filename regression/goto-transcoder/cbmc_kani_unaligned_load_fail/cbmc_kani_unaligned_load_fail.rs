// The .goto is `kani unaligned_load_fail.rs --only-codegen --keep-temps` of this file saved as unaligned_load_fail.rs (Kani 0.68.0, CBMC 6.11.0).
// Negative: the unaligned load still reads bytes 1..4, so a wrong shift fails.
#![feature(core_intrinsics)]
#![allow(internal_features)]

#[kani::proof]
fn harness() {
    let buf: [u32; 2] = kani::any();
    let p = unsafe { (buf.as_ptr() as *const u8).add(1) } as *const u32;
    let v = unsafe { std::intrinsics::unaligned_volatile_load(p) };
    assert!(v == (buf[0] >> 8) | (buf[1] << 16));
}
