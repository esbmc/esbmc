// The .goto is `kani github_10936_iter.rs --only-codegen --keep-temps` (Kani 0.68.0, CBMC 6.11.0).
// A byte slice over a local, walked by slice::Iter, must keep pointing at the local (#10936).
#[kani::proof]
#[kani::unwind(5)]
fn harness() {
    let n: u32 = kani::any();
    let raw = &n as *const u32 as *const u8;
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(raw, core::mem::size_of::<u32>()) };
    let is_zero = bytes.iter().all(|&b| b == 0);
    assert!(is_zero == (n == 0));
}
