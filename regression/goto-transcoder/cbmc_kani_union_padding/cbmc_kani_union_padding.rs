// The .goto is `kani union_pad.rs --only-codegen --keep-temps` of this file saved as union_pad.rs (Kani 0.68.0, CBMC 6.11.0).
// A union variant narrower than the union: Kani pads it to the union's size.
union Word {
    byte: u8,
    word: u32,
}

#[kani::proof]
fn harness() {
    let v: u8 = kani::any();
    kani::assume(v < 10);
    let small = Word { byte: v };
    assert!(unsafe { small.byte } < 10);
    assert!(unsafe { small.word } & 0xff == v as u32);
}
