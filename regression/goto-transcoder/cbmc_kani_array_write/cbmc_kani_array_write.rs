// The .goto is `kani arr_write.rs --only-codegen --keep-temps` of this file saved as arr_write.rs (Kani 0.68.0, CBMC 6.11.0).
// A whole-array store through a pointer to a differently shaped array.
#[kani::proof]
fn harness() {
    let mut halves = [0u16; 2];
    unsafe { *(&mut halves as *mut [u16; 2] as *mut [u8; 4]) = [1, 2, 3, 4] };
    assert!(halves[0] == 0x0201);
}
