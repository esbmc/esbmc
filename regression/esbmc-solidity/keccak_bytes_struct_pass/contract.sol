// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression for keccak256 / sha256 fallback when the argument is a
// raw source-level bytes value (BytesDynamic / BytesStatic struct).
// Previously crashed symex with a uint256 → BytesStatic struct
// mismatch when the function's return value was assigned into a
// bytes32 slot.  The frontend now emits a typed nondet bytes32
// directly when the input is a bytes struct.
contract C {
    string public s;
    bytes public bd;
    function setS(string calldata _s) public { s = _s; }
    function setBd(bytes calldata _b) public { bd = _b; }
    function hashString() public view returns (bytes32) {
        return keccak256(bytes(s));
    }
    function hashBytes() public view returns (bytes32) {
        return keccak256(bd);
    }
    function sha() public view returns (bytes32) {
        return sha256(bytes(s));
    }
}

contract H {
    function check(string calldata _s, bytes calldata _b) public {
        C c = new C();
        c.setS(_s);
        c.setBd(_b);
        bytes32 a = c.hashString();
        bytes32 b = c.hashBytes();
        bytes32 d = c.sha();
        assert(a == a); assert(b == b); assert(d == d);
    }
}
