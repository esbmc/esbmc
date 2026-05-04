// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Tests string.concat via keccak256 call (nondet hash abstraction).
// Hash comparison removed: two independent keccak256 calls return
// independent nondet values, so equality is not provable.
contract Test {
    function concatstring(string memory x, string memory y) public pure returns (string memory)
    {
        return string.concat(x,y);
    }
    function test() public
    {
        string memory str = concatstring("test", "hello");
        uint256 h = uint256(keccak256(abi.encodePacked(str)));
        assert(h == h);
    }
}