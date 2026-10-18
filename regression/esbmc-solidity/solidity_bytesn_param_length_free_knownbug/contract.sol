// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// KNOWNBUG: an ABI bytesN argument's `.length` is not pinned.
//
// bytesN lowers to `BytesStatic { unsigned char data[32]; size_t length; }`
// (solidity_bytes.c:16-19). The `.length` field is legitimate -- N ranges over
// [1,32] -- but the harness builds a public function's arguments in
// `assign_param_nondet` (solidity_convert_call.cpp), where bytesN falls into
// the branch whose comment calls it a "Scalar harness parameter", and is handed
// a bare `sideeffect(nondet)` of the whole struct. That leaves `.length` as
// free as `.data`.
//
// `bytes_static_equal` returns FALSE OUTRIGHT when the two lengths differ
// (solidity_bytes.c:354), before comparing a single byte. On the chain there is
// no calldata encoding of a bytesN with any length but N, so the not-equal arm
// is reachable here for a reason no transaction can produce.
//
// `roundTrip` below is a Solidity tautology: converting a bytes32 to uint256
// and back is the identity. This test PINS THE DEFECT -- it currently reports
// VERIFICATION FAILED, and when the length is pinned it will report SUCCESSFUL
// and this test will flip, which is the point of recording it.
//
// WHY IT IS A knownbug AND NOT A FIX. Pinning the length via
// `bytes_static_from_uint(nondet_uint256(), N)` was implemented and MEASURED:
// it makes this tautology SUCCESSFUL, and it routes around `get_nondet_expr`,
// whose minted `nondet$symex::` symbol is the only form the counterexample
// harvest can read back. The parameter's value became unrecoverable, the
// emitter reported it DEFAULTED, and the generated-test funnel on
// notes/coverage/poc/D11_Bytes32Equality.sol went from 3 paths / 3 cases /
// 0 merged to 3 paths / 1 case / 1 merged. A sound model whose counterexample
// cannot be read back is not an improvement for a pipeline whose deliverable is
// the test, so it was reverted. A real fix must pin the length AND keep the
// value recoverable.
//
// The companion PoC with all three probes (including the two that already
// behave correctly, so a future fix is not mistaken for a repair of everything)
// is notes/coverage/poc/D12_Bytes32LengthFree.sol.
contract C
{
  function roundTrip(bytes32 b) public pure
  {
    assert(b == bytes32(uint256(b)));
  }
}
