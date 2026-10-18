// A local variable declared WITH an initialiser inside a function that is
// INHERITED by the contract under verification must keep that initialiser.
//
// merge_inheritance_ast()'s add_inherit_label() stamps `is_inherited` on every
// sub-node that carries an `id`, which includes locals inside an inherited
// function body.  get_var_decl() serves both state-variable declarations and
// `rule variable-declaration-statement`, and used to suppress the initialiser
// for anything marked inherited -- a rule that is only correct for state
// variables (whose initial value is replayed by move_inheritance_to_ctor()).
// The consequence was that `uint256 y = x + 1;` below silently became `y = 0`.
//
// `s` is here to pin the other half: an inherited STATE variable with an
// initialiser must still arrive at the derived contract holding its value.
pragma solidity ^0.8.0;

contract B {
    uint256 internal s = 42;

    function f(uint256 x) internal pure returns (uint256) {
        uint256 y = x + 1;
        return y;
    }
}

contract D is B {
    uint256 public out;

    function g(uint256 x) public {
        if (x > 1000) return;
        out = f(x);
        assert(out == x + 1);
        assert(s == 42);
    }
}
