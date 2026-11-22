// The .goto is `kani nonnull_write.rs --only-codegen --keep-temps` of this file saved as nonnull_write.rs (Kani 0.68.0, CBMC 6.11.0).
// NonNull::new_unchecked reads a pointer that is not a constant address as a NonNull with a byte_extract (#10889).
use std::ptr::NonNull;

#[kani::proof]
fn harness() {
    let mut a: u32 = kani::any();
    let mut b: u32 = kani::any();
    let raw: *mut u32 = if kani::any() { &mut a } else { &mut b };
    let p = NonNull::new(raw).unwrap();
    let v: u32 = kani::any();
    unsafe { p.as_ptr().write(v) };
    assert!(a == v || b == v);
}
