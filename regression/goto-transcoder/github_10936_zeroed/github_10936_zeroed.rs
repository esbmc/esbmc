// The .goto is `kani github_10936_zeroed.rs -Z function-contracts --only-codegen --keep-temps` (Kani 0.68.0, CBMC 6.11.0).
// The shape of verify-rust-std's MaybeUninit::zeroed contract, checked by check_zeroed_* (#10936).
use std::mem::MaybeUninit;

#[kani::ensures(|result: &MaybeUninit<u32>| {
    let bytes = unsafe {
        core::slice::from_raw_parts(result.as_ptr() as *const u8, core::mem::size_of::<u32>())
    };
    bytes.iter().all(|b| *b == 0)
})]
fn zeroed() -> MaybeUninit<u32> {
    let mut u = MaybeUninit::<u32>::uninit();
    unsafe { u.as_mut_ptr().write_bytes(0u8, 1) };
    u
}

#[kani::proof_for_contract(zeroed)]
fn harness() {
    let m = zeroed();
    let v = unsafe { m.assume_init() };
    assert!(v == 0);
}
