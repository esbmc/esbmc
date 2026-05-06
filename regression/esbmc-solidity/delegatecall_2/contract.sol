// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

contract Logic {
    uint public x;

    function setX(uint _x) public {
        x = _x;
    }
}

contract Proxy {
    uint public x;

    function test(address logic) public {
        (bool success, bytes memory data) = logic.delegatecall(
            abi.encodeWithSignature("setX(uint256)", 42)
        );
        // should fail: success cannot be guaranteed
        assert(success);
    }
}
