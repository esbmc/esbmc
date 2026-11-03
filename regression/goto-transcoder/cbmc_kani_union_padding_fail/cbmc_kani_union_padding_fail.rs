// The .goto is `kani union_pad_fail.rs --only-codegen --keep-temps` of this file saved as union_pad_fail.rs (Kani 0.68.0, CBMC 6.11.0).
// A union variant narrower than the union: Kani pads it to the union's size,
// and the padding bytes are unconstrained.
union Word {
    byte: u8,
    word: u32,
}

#[kani::proof]
fn harness() {
    let small = Word { byte: 5 };
    assert!(unsafe { small.word } == 5);
}
