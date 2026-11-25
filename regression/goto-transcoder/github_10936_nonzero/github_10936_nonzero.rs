// The .goto is `kani github_10936_nonzero.rs -Z function-contracts --only-codegen --keep-temps` (Kani 0.68.0, CBMC 6.11.0).
// The shape of verify-rust-std's NonZero::new contract, checked by nonzero_check_new_for_* (#10936).
use std::num::NonZero;

#[kani::ensures(|result: &Option<NonZero<u32>>| {
    let size = core::mem::size_of::<u32>();
    let n_ptr: *const u32 = &n;
    let n_slice = unsafe { core::slice::from_raw_parts(n_ptr as *const u8, size) };
    let n_is_zero = n_slice.iter().all(|&byte| byte == 0);
    let created_iff_nonzero = result.is_some() == !n_is_zero;
    let value_preserved = match result {
        Some(nz) => {
            let inner: u32 = nz.get();
            let inner_ptr: *const u32 = &inner;
            let inner_slice = unsafe { core::slice::from_raw_parts(inner_ptr as *const u8, size) };
            n_slice == inner_slice
        }
        None => true,
    };
    created_iff_nonzero && value_preserved
})]
fn mk(n: u32) -> Option<NonZero<u32>> {
    NonZero::new(n)
}

#[kani::proof_for_contract(mk)]
fn harness() {
    let x: u32 = kani::any();
    let _ = mk(x);
}
