// The .goto is `kani misaligned_deref_fail.rs --only-codegen --keep-temps` of this file saved as misaligned_deref_fail.rs (Kani 0.68.0, CBMC 6.11.0).
// Negative: a plain dereference keeps Kani's own alignment assertion, which
// ESBMC still checks on a CBMC goto-binary (#10892).
#[kani::proof]
fn harness() {
    let buf: [u32; 2] = kani::any();
    let p = unsafe { (buf.as_ptr() as *const u8).add(1) } as *const u32;
    let _v = unsafe { *p };
}
