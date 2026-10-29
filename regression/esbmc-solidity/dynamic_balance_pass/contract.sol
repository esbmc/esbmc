// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for cpp_new lowering: when `new C{value:V}()` is followed by
// reads of `address(...).balance` for the dynamic instance, the runtime
// contract-instance registry must contain a pointer to the actual heap
// allocation, not a dead stack local. Prior to the fix in
// solidity_convert_constructor.cpp::get_new_object_ctor_call, the lowering
// emitted `Vault(&tmp$N); *new_ptr = tmp$N` so the constructor's
// `_ESBMC_get_unique_address(this, cname)` registered `&tmp$N` (a dead
// local), and any subsequent runtime lookup of address(.).balance returned
// the value that tmp$N had at construction time, ignoring all later
// `transfer/send` debits.
//
// Routing through `address a = address(v); a.balance` defeats the
// frontend's CONTRACT-typed short-circuit and forces the lookup through
// `get_aux_property_function`, which is exactly the path the fix repairs.

contract Vault {
    constructor() payable {}
    function withdraw(address payable to, uint256 amt) external {
        to.transfer(amt);
    }
}

contract Probe {
    function check() external payable {
        Vault v = new Vault{value: 100}();
        address a = address(v);   // launder through `address` type
        uint pre  = a.balance;
        v.withdraw(payable(address(0x1234)), 30);
        uint post = a.balance;
        assert(pre == post + 30);
    }
}
