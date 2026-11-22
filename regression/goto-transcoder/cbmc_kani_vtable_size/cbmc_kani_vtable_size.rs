// The .goto is `kani vtable_size.rs --only-codegen --keep-temps` of this file (Kani 0.68.0, CBMC 6.11.0).
// Transmuting DynMetadata to *const () reads the vtable pointer as another pointer type (#10889).
#![feature(ptr_metadata, core_intrinsics)]
trait Value {
    fn value(&self) -> u32;
}

impl Value for u32 {
    fn value(&self) -> u32 {
        *self
    }
}

#[kani::proof]
fn harness() {
    let x: u32 = kani::any();
    let fat: *const dyn Value = &x;
    let vptr = unsafe {
        std::mem::transmute::<std::ptr::DynMetadata<dyn Value>, *const ()>(std::ptr::metadata(fat))
    };
    assert!(unsafe { std::intrinsics::vtable_size(vptr) } == 4);
    assert!(unsafe { std::intrinsics::vtable_align(vptr) } == 4);
}
