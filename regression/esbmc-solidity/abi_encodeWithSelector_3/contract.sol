// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Test abi.encodeWithSelector: nondet selector should produce nondet encoding.
contract AbiEncodeWithSelectorDiff {
    function test(bytes4 sel) public pure {
        uint256 h = uint256(keccak256(abi.encodeWithSelector(sel, uint256(1))));
        assert(h == 0); // must FAIL: nondet selector => nondet hash
    }
}
