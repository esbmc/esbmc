// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: a user-defined `returns (bytes32)` function whose body is
// `keccak256(abi.encodePacked(...))` used to crash symex with
// "Looking up index of nonexistant member \"data\" in struct/union
// \"index\"" (value_sett::make_member, struct/scalar shape mismatch).
// Root cause: the frontend only packed keccak/sha results into the
// BytesStatic struct when the inner arg was a raw bytes value, so an
// `abi.encodePacked` arg kept the uint256 library-call type while the
// function signature required bytes32.  The outer assignment then
// drove value-set assignment across incompatible types.
//
// Fix (src/solidity-frontend/solidity_convert_expr.cpp): always pack
// keccak256 / sha256 results into bytes32, regardless of inner arg
// shape.  Pack/unpack is identity for length 32 so functional
// consistency is preserved.
contract C {
    mapping(bytes32 => string) public docs;

    function submit(string memory s) public {
        bytes32 h = getHash(s);
        docs[h] = s;
    }

    function getHash(string memory s) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(s));
    }

    function getHashSha(string memory s) public pure returns (bytes32) {
        return sha256(abi.encodePacked(s));
    }
}

contract T {
    function test(string memory s) public {
        C c = new C();
        c.submit(s);
        bytes32 a = c.getHash(s);
        bytes32 b = c.getHash(s);
        // Functional consistency: same input -> same hash.
        assert(a == b);
    }
}
