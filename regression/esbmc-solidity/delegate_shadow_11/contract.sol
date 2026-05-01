// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// encodeCall form: the function reference is the first argument and the
// second argument is a tuple of user args. encodeCall is type-checked by
// solc at compile time so arg count/types must match the referenced
// function; that same information is what the shadow path uses to dispatch.

contract Logic {
    uint256 public x;
    uint256 public y;
    function setBoth(uint256 a, uint256 b) public {
        x = a;
        y = b;
    }
}

contract Proxy {
    uint256 public x;
    uint256 public y;

    function test() public {
        x = 0;
        y = 0;
        address(this).delegatecall(
            abi.encodeCall(Logic.setBoth, (uint256(10), uint256(20)))
        );
        assert(x == 10);
        assert(y == 20);
    }
}
