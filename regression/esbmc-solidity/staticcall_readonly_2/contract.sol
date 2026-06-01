// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

// Bound-mode staticcall read-only: same scenario, but the post-call
// assertion claims the opposite ("state MUST have changed"). Since
// snapshot/restore in $staticcall#0 rolls back any writes the target
// performs, the claim fails as expected.
contract Target {
    uint public x;

    function getX() public view returns (uint) {
        return x;
    }

    function modify(uint v) public {
        x = v;
    }
}

contract Caller {
    Target t = new Target();

    function test() public {
        t.modify(100);
        uint before_x = t.x();
        (bool success, ) = address(t).staticcall(abi.encodeWithSignature("getX()"));
        uint after_x = t.x();
        assert(before_x != after_x);
        assert(success || !success);
    }
}
