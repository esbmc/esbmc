// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

// FAIL dual of yul_struct_reinterpret_pass: documents the over-
// approximation contract. The bytes32<->struct inline-assembly
// reinterpret is havoc'd (nondet of the declared type), so the value
// threaded through the function-pointer context is NOT recoverable.
// Asserting it equals the originally-installed result must therefore
// FAIL: the fix is structurally sound (no crash, well-typed) but
// value-imprecise by construction.
contract C
{
  struct Info
  {
    function() internal view returns (uint256) getTotalSupply;
    bytes32 dataSlot;
  }

  function _ts() private pure returns (uint256)
  {
    return 7;
  }

  function _contextToInfo(bytes32 ctx)
    private
    pure
    returns (Info memory self)
  {
    assembly ("memory-safe") {
      self := ctx
    }
  }

  function _infoToContext(Info memory self)
    private
    pure
    returns (bytes32 ctx)
  {
    assembly ("memory-safe") {
      ctx := self
    }
  }

  function _lazyGetSupply(bytes32 ctx) private view returns (uint256)
  {
    Info memory self = _contextToInfo(ctx);
    return self.getTotalSupply();
  }

  function run() external view
  {
    Info memory i;
    i.getTotalSupply = _ts;
    bytes32 c = _infoToContext(i);
    // _ts() returns 7, but the reinterpret is over-approximated to
    // nondet, so this is NOT provable -> VERIFICATION FAILED.
    assert(_lazyGetSupply(c) == 7);
  }
}
