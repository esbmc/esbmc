// The .goto is `kani report_fail.rs --only-codegen --keep-temps` of this file saved as report_fail.rs (Kani 0.68.0 at 0aac16e, CBMC 6.11.0).
// The same shape with an assertion that fails for x == 255.
#[kani::proof]
fn harness() {
    let x: u8 = kani::any();
    let zero: u8 = 0;
    kani::cover!(x > 5);
    kani::cover!(x > 5 && x < 3);
    kani::cover!(zero > 1);
    if x > 5 && x < 3 {
        assert!(x == 0);
    }
    if zero > 1 {
        assert!(x == 1);
    }
    assert!(x < 255);
}
