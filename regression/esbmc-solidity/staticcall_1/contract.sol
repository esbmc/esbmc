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
        // In unbound mode, success is nondet
        // Just check the call doesn't crash
        assert(success || !success);
    }
}
