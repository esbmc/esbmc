// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dynamic-array / bytes calldata argument rendering + correct method
// attribution in a multi-function dispatcher. `feed` takes `bytes calldata` and
// `address[] calldata`; the generator must render `hex""` and `new address[](4)`
// (length 4 = the external-call harness model) rather than degrading to
// UNSUPPORTED, and must attribute the covered branch to `feed` (not to the
// other dispatcher method `poke`) — a regression for the whole-unit
// wrong-method reconstruction.
contract A {
    uint256 public s;
    function poke(uint256 k) external { if (k == 7) s = 1; }
    function feed(bytes calldata b, address[] calldata xs) external {
        if (xs.length != 0) s = 2;
    }
}
