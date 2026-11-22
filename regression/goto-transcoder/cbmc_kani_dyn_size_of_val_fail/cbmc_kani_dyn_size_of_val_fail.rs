// The .goto is `kani dyn_size_fail.rs --only-codegen --keep-temps` of this file saved as dyn_size_fail.rs (Kani 0.68.0, CBMC 6.11.0).
// As cbmc_kani_dyn_size_of_val, with a wrong expected size (#10889).
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
    assert!(std::mem::size_of_val(d) == 8);
    assert!(std::mem::align_of_val(d) == 4);
}
