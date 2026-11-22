// The .goto is `kani unaligned_struct_store_fail.rs --only-codegen --keep-temps` of this file saved as unaligned_struct_store_fail.rs (Kani 0.68.0, CBMC 6.11.0).
// Negative: a misaligned store into a struct field is still modelled, so the
// byte it writes is seen (#10892).
#[repr(C)]
struct S { x: u32, y: u32, z: u32, w: u32 }
#[repr(C, packed)]
struct P { a: u8, b: u32 }

#[kani::proof]
fn harness() {
    let mut s = S { x: 0, y: 0, z: 0, w: 0 };
    let k: usize = kani::any();
    kani::assume(k <= 1);
    let p = unsafe { (&mut s as *mut S as *mut u8).add(k) } as *mut P;
    unsafe { (*p).b = 0x1122_3344 };
    let b = unsafe { *(&s as *const S as *const u8).add(k + 1) };
    assert!(b == 0);
}
