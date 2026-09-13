pragma solidity >=0.8.0;

contract Contract {
    string s = "HELLO";

    function test() public {
        string memory s_local = "HELLO";
        s = "WORLD";
        s_local = "WORLD";
    }
}
