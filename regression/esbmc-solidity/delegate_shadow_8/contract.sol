// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Nested helper inlining with return values: Logic.run() -> _outer() ->
// _inner(). Each helper carries a return value that flows through the
// outer helper back into the state write. Proxy layout is swapped to
// rule out coincidental struct-offset agreement.

contract Logic {
    uint256 public x;
    uint256 public y;

    function _inner(uint256 v) internal returns (uint256) {
        y = v;
        return v + 1;
    }

    function _outer(uint256 v) internal {
        x = _inner(v);
    }

    function run(uint256 v) public {
        _outer(v);
    }
}

contract Proxy {
    uint256 public y;
    uint256 public x;

    function test() public {
        x = 0;
        y = 0;
        address(this).delegatecall(
            abi.encodeWithSignature("run(uint256)", uint256(41))
        );
        assert(y == 41);
        assert(x == 42);
    }
}
