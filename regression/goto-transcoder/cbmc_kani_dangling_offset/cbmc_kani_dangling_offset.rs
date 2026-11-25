// The .goto is `kani dangling_offset.rs --only-codegen --keep-temps` of this file saved as dangling_offset.rs (Kani 0.68.0, CBMC 6.11.0).
// NonNull::dangling() is the integer 4; no object may sit there, as in CBMC, or the offset pointer names it (#10905).
use std::ptr::NonNull;

#[kani::proof]
fn harness() {
    let buf: [u32; 1] = kani::any();
    let ptr1: *const u32 = buf.as_ptr();
    let off: usize = kani::any_where(|b: &usize| *b < 4);
    let ptr2: *const u32 = NonNull::<u32>::dangling().as_ptr().wrapping_add(off);
    if ptr1 as isize == ptr2 as isize {
        assert!(ptr1 == ptr2);
    }
}
