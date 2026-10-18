// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dual of send_callee_revert_fail: `.send`'s `ok = !reverted` tracks the
// callee outcome, so `ok ==> credited == before + 1` holds (ok==false means
// the receive reverted and its write was rolled back).
contract C {
    bool reject;
    uint256 credited;
    constructor(bool r) { reject = r; }
    receive() external payable { require(!reject, "no"); credited += 1; }
    function test() public {
        require(address(this).balance >= 1);
        uint256 before = credited;
        bool ok = payable(address(this)).send(1);
        assert(!ok || credited == before + 1);
    }
}
