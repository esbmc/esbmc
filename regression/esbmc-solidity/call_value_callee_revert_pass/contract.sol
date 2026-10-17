// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual of call_value_callee_revert_fail.  The low-level `.call{value:}`
// success flag is `ok = !reverted`, so it tracks the callee outcome
// precisely rather than being a constant nondet/true:
//   * ok == true   <=>  the receive completed and its effect is visible
//                       (credited bumped by one);
//   * ok == false  <=>  the receive reverted and its write was rolled back
//                       (credited unchanged).
// Hence `ok ==> credited == before + 1` holds, which a naive "ok is always
// nondet" model would refute.  Self-call hits the tracked singleton path.
contract C {
    bool reject;
    uint256 credited;
    constructor(bool r) { reject = r; }

    receive() external payable {
        require(!reject, "rejected");
        credited += 1;
    }

    function test() public {
        require(address(this).balance >= 1);
        uint256 before = credited;
        (bool ok, ) = payable(address(this)).call{value: 1}("");
        assert(!ok || credited == before + 1);
    }
}
