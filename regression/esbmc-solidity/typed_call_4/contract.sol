// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Overload disambiguation: the contract has two functions both named
// `poke` but with different parameter types. Signature-based dispatch
// must route to the right overload based on the signature string.
contract TypedCallOverload {
    uint256 public u;
    address public a;

    function poke(uint256 v) public { u = v; }
    function poke(address x) public { a = x; }

    function test() public {
        u = 0;
        a = address(0);
        // Route to poke(uint256) — must write u, not a.
        address(this).call(abi.encodeWithSignature("poke(uint256)", uint256(77)));
        assert(u == 77);
        assert(a == address(0));
    }
}
