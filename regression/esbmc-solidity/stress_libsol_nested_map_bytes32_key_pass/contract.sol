// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for get_expr / IndexAccess nested-mapping path using a raw
// solidity_gen_typecast when the outer index resolution left the key as
// a BytesStatic struct. xor_fold_key_to_64bit then built
// shr(BytesStatic, uint256), which migrate_expr cannot lower, aborting
// with "migrate expr failed". The fix routes the nested-path cast
// through gen_mapping_key_typecast so bytesN keys go through
// bytes_static_to_mapping_key.
//
// Write and read in the same transaction so the harness over-approx
// cannot invalidate the invariant.

contract C {
    mapping(address => mapping(bytes32 => uint256)) private store;

    function go() external {
        address a = address(0x1);
        bytes32 k = bytes32(uint256(1));
        store[a][k] = 42;
        assert(store[a][k] == 42);
    }
}
