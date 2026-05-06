// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

contract Target {
    uint public x;

    function getX() public view returns (uint) {
        return x;
    }
}

contract Caller {
    function test(address target) public {
        (bool success, bytes memory data) = target.staticcall(
            abi.encodeWithSignature("getX()")
        );
        // should fail: success cannot be guaranteed
        assert(success);
    }
}
