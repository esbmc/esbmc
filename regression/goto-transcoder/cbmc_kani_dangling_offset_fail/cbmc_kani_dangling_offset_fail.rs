// The .goto is `kani dangling_offset_fail.rs --only-codegen --keep-temps` of this file saved as dangling_offset_fail.rs (Kani 0.68.0, CBMC 6.11.0).
// A pointer made from an integer above the first page can still share an object's address (#10905).
#[kani::proof]
fn harness() {
    let buf: [u32; 1] = kani::any();
    let ptr1: *const u32 = buf.as_ptr();
    let addr: usize = kani::any();
    let ptr2: *const u32 = std::ptr::without_provenance(addr);
    assert!(ptr1 as usize != ptr2 as usize);
}
