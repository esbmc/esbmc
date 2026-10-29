// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// P3 regression-lock: nested-mapping(K=>V[]) pop on length=0 (path at
// solidity_convert_ref.cpp:957) — same direct-decrement underflow as
// the state-var path.
//
// KNOWNBUG until S3 lands.
contract C {
    mapping(address => uint[]) m;

    function f(address k) public {
        require(m[k].length == 0);
        m[k].pop();
        assert(m[k].length < 1000000);
    }
}
