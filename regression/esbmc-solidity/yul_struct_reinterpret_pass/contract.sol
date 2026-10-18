// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

// Minimal reproducer for the 1inch FarmingLib idiom that packs a memory
// struct pointer into a bytes32 (to thread it through a function-pointer
// callback context) via inline assembly. The bytes32<->struct reinterpret
// collapses the receiver type and aborts on the member lookup of the
// function-pointer member. Pinned KNOWNBUG; flips to CORE once the
// Solidity-frontend Yul reinterpret is over-approximated soundly.
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

  function run() external view returns (uint256)
  {
    Info memory i;
    i.getTotalSupply = _ts;
    bytes32 c = _infoToContext(i);
    return _lazyGetSupply(c);
  }
}
