pragma solidity >=0.8.0;
// Exercises the underflow guard: when the symbolic balance is less
// than the value argument, the transfer would have reverted in real
// EVM, so the model's pre - val underflow path stays unreachable
// thanks to the `__re_drain_val > __re_drain_pre` LHS of the
// disjunction in the assert.  Without that guard, the path would
// produce a spurious counterexample.
contract Pay {
    function pay(address payable to, uint256 amt) external {
        require(amt > 0);
        to.transfer(amt);
    }
}
