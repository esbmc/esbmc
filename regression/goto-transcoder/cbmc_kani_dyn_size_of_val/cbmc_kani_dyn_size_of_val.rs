// The .goto is `kani dyn_size.rs --only-codegen --keep-temps` of this file saved as dyn_size.rs (Kani 0.68.0, CBMC 6.11.0).
// Kani reads the vtable pointer out of DynMetadata with a byte_extract (#10889).
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
    let d: &dyn Value = &x;
    assert!(std::mem::size_of_val(d) == 4);
    assert!(std::mem::align_of_val(d) == 4);
}
