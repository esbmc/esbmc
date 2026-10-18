// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// FAIL dual of modifier_named_tuple_return_pass.  The named return
// parameter `result` is correctly visible inside the modifier-wrapped
// body now, so the assertion that checks a user-writable invariant on
// it reaches the solver and is refutable.
contract C {
    uint256 public threshold;

    modifier overThreshold(uint256 v) {
        require(v > threshold, "below");
        _;
    }

    // Named return parameters + modifier — before F1 this crashed
    // symex before the body executed.  Now the body runs and its
    // written value flows to the caller; we assert a constraint that
    // the caller *can* violate.
    function compute(uint256 v) public overThreshold(v) view
        returns (uint256 result)
    {
        result = v - threshold;
    }
}

contract H {
    function test(uint256 v) public {
        C c = new C();
        uint256 r = c.compute(v);
        // With overThreshold(v) requiring v > threshold, and result =
        // v - threshold, result is in (0, 2^256 - threshold). Asserting
        // result == 0 is refutable.
        assert(r == 0);
    }
}
