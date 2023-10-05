// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract D {
    function zero() public view returns (uint)  {
        return 0;
    }
}

contract C {
    uint x;
	function f(D d) public view {
        uint res = d.zero();
        assert(x < 1000);
	}

    function inc() public {
        ++x;
    }
}

