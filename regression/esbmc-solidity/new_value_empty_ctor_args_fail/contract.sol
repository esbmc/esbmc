// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative counterpart: same shape, wrong expected balance.
contract Bal {
    constructor() payable {}
    function balanceOf() public view returns (uint) {
        return address(this).balance;
    }
}

contract Caller {
    function deploy() public {
        Bal b = new Bal{value: 100}();
        // Wrong: balance is 100, not 200 — assert must fail.
        assert(b.balanceOf() == 200);
    }
}
