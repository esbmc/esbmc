// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

contract SliceTest {
    function test(bytes calldata data) external pure {
        if (data.length >= 4) {
            bytes calldata first4 = data[:4];
            // slice is nondet (over-approximation), length >= 0 is tautology
            assert(first4.length >= 0);
        }
        // no crash, conversion succeeds
        assert(true);
    }
}
