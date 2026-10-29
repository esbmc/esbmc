// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Parent constructor declares a local variable that is then assigned to a
// state variable. When a derived contract is parsed, the parent body is
// re-walked and get_var_decl re-encounters the same VariableDeclaration —
// the early-return path used to leak a bare symbol_exprt as the statement,
// which then tripped `goto_symext: unexpected statement: symbol`.
contract Parent {
    address public owner;

    constructor() {
        address tmp = msg.sender;
        owner = tmp;
    }
}

contract Child is Parent {
    function check() public view {
        assert(owner == owner);
    }
}
