// The .goto is `kani arr_write_wrong.rs --only-codegen --keep-temps` of this file saved as arr_write_wrong.rs (Kani 0.68.0, CBMC 6.11.0).
// As arr_write.rs, but expects the wrong stored value.
#[kani::proof]
fn harness() {
    let mut halves = [0u16; 2];
    unsafe { *(&mut halves as *mut [u16; 2] as *mut [u8; 4]) = [1, 2, 3, 4] };
    assert!(halves[0] == 0x0202);
}
