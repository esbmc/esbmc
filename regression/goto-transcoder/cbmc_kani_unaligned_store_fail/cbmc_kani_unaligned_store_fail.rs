// The .goto is `kani unaligned_store_fail.rs --only-codegen --keep-temps` of this file saved as unaligned_store_fail.rs (Kani 0.68.0, CBMC 6.11.0).
// The stored value reads back, so a wrong expected value must fail on its assertion (#10891).
#![feature(core_intrinsics)]
#![allow(internal_features)]
use std::mem::MaybeUninit;

#[kani::proof]
fn harness() {
    let mut buffer = [MaybeUninit::<u32>::uninit(); 4];
    let base = buffer.as_mut_ptr() as *mut u8;
    let off: usize = kani::any_where(|o: &usize| *o <= 12);
    let dst = base.wrapping_add(off) as *mut u32;
    let val: u32 = kani::any();
    unsafe { std::intrinsics::unaligned_volatile_store(dst, val) };
    let read_back = unsafe { dst.read_unaligned() };
    assert!(read_back == val.wrapping_add(1));
}
