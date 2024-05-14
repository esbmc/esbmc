// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Test {
    function concatstring(string memory x, string memory y) public pure returns (string memory)
    {
        return string.concat(x,y);
    }
    function test() public
    {
        string memory str = concatstring("test", "hello");
        assert(sha256(abi.encodePacked(str)) == sha256(abi.encodePacked("testhello")));
    }
}