// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: `mapping(K => T[])` where T is a STRUCT (containing at
// least one non-uint256 field — e.g. a `bytes32` that maps to
// BytesStatic after the keccak pack fix 31106af1c5) used to crash
// symex at the push call:
//   function call: argument "_ESBMC_array_push_uint256@element"
//   type mismatch: got struct, expected unsignedbv
// The frontend hard-coded `_ESBMC_array_push_uint256` for the
// mapping-of-dynarray write-through regardless of element type, so
// struct values violated the helper's uint256 element signature at
// the GOTO call binding.  Reproduced on SolidiFi buggy_46's
// `mapping(address => FileExistenceStruct[])` with a `bytes32`
// QRCodeHash field.
//
// Fix: dispatch on element type in
// src/solidity-frontend/solidity_convert_ref.cpp — keep the
// uint256-specialised typed helper for `uint256[]` elements, and
// fall back to the generic `_ESBMC_array_push(array, &elem, sizeof)`
// for every other element type (structs, bytes32 fields, smaller
// integers, etc.).
contract C {
    struct FileInfo {
        uint256 date;
        address sender;
        bytes32 hash;   // BytesStatic under our model
    }

    mapping(address => FileInfo[]) public files;

    function register(bytes32 h) public {
        FileInfo memory info;
        info.date = block.timestamp;
        info.sender = msg.sender;
        info.hash = h;
        files[msg.sender].push(info);
    }
}

contract H {
    function test(bytes32 h) public {
        C c = new C();
        c.register(h);
    }
}
