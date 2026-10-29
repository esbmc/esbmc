// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pin-test: abi.decode is detached nondet (ledger #4). The
// round-trip `decode(encode(x)) == x` invariant does NOT hold
// under the current modelling. Pin via planted assert.
contract C {
    function test(uint256 x) public {
        bytes memory enc = abi.encode(x);
        uint256 y = abi.decode(enc, (uint256));
        assert(y == x); // must FAIL: y is detached nondet from x
    }
}
