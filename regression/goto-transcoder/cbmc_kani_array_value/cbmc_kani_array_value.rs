// The .goto is `kani arr_value.rs --only-codegen --keep-temps` of this file saved as arr_value.rs (Kani 0.68.0, CBMC 6.11.0).
// Whole-array loads through a pointer: the same type at offset 0, a scalar
// of the same width, a reinterpreted array, and a struct with padding.
#[repr(C)]
struct Padded {
    a: u8,
    b: u32,
}

#[kani::proof]
fn harness() {
    let bytes: [u8; 4] = [1, 2, 3, kani::any()];
    let copy: [u8; 4] = unsafe { *(&bytes as *const [u8; 4]) };
    assert!(copy[0] == 1 && copy[2] == 3 && copy[3] == bytes[3]);

    let word: u32 = 0x0403_0201;
    let le: [u8; 4] = unsafe { *(&word as *const u32 as *const [u8; 4]) };
    assert!(le[0] == 1 && le[3] == 4);

    let halves: [u16; 2] = [0x0201, 0x0403];
    let b: [u8; 4] = unsafe { *(&halves as *const [u16; 2] as *const [u8; 4]) };
    assert!(b[1] == 2 && b[2] == 3);

    let s = Padded { a: 7, b: 9 };
    let raw: [u8; 8] = unsafe { *(&s as *const Padded as *const [u8; 8]) };
    assert!(raw[0] == 7 && raw[4] == 9);
}
