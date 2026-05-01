// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Adversarial counterpart to delegate_shadow_5: same early-return body,
// but the post-delegatecall assertion is deliberately wrong.  Must
// produce VERIFICATION FAILED, proving both that the return-rewrite
// did not silently accept any value AND that the early return did not
// let the caller skip the assertion.

contract Logic {
    uint256 public x;
    function earlyExit(uint256 v) public returns (uint256) {
        if (v == 7) {
            x = 100;
            return 999;
        }
        x = v;
        return v;
    }
}

contract Proxy {
    uint256 public x;

    function test() public {
        x = 0;
        address(this).delegatecall(
            abi.encodeWithSignature("earlyExit(uint256)", uint256(7))
        );
        // After the delegatecall x must be 100 (v==7 took the early
        // branch). Asserting == 42 instead must FAIL — the property
        // has to be evaluated after the delegatecall, not skipped by
        // the target's return statement.
        assert(x == 42);
    }
}
