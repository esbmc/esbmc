// The .goto is `kani dyn_call.rs --only-codegen --keep-temps` of this file saved as dyn_call.rs (Kani 0.68.0, CBMC 6.11.0).
// A dyn call goes through a vtable that Kani initialises with a statement
// expression; under --function those initialisers must still run.
trait Value {
    fn value(&self) -> u32;
}

struct Seven;

impl Value for Seven {
    fn value(&self) -> u32 {
        7
    }
}

#[kani::proof]
fn harness() {
    let s = Seven;
    let v: &dyn Value = &s;
    assert!(v.value() == 7);
}
