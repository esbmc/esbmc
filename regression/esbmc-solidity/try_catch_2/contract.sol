// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract Target {
    function getValue() external pure returns (uint) {
        return 42;
    }
}

contract TryCatchFail {
    Target target;
    uint public result;
    bool public ok;

    constructor() {
        target = new Target();
    }

    function test() public {
        try target.getValue() returns (uint v) {
            result = v;
            ok = true;
        } catch {
            result = 0;
            ok = false;
        }
        // FAIL: in the catch branch, ok is false
        assert(ok == true);
    }
}
