// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// P3 regression-lock: `mapping(K=>V)[]` pop on length=0 (path at
// solidity_convert_ref.cpp:815, `#sol_mapping_array` flag).
//
// KNOWNBUG until S3 lands.
contract C {
    mapping(uint => uint)[] arr;

    function f() public {
        require(arr.length == 0);
        arr.pop();
        assert(arr.length < 1000000);
    }
}
