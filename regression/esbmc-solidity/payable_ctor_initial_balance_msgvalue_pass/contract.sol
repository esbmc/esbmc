// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Audit finding F7: `new C{value: V}()` with payable constructor must
// deploy with $balance == msg.value (= V). Today this is correctly
// modelled (balance_init = msg_value). Lock as CORE PASS so S1.1's
// gen_zero fix to the non-payable branch doesn't accidentally regress
// this path.
contract Vault {
    constructor() payable {}
}

contract H {
    Vault v;
    constructor() payable {
        require(msg.value >= 100);
        v = new Vault{value: 100}();
    }
    function check_balance_eq_value() public view {
        assert(address(v).balance == 100);
    }
}
