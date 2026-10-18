// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Audit finding F4: _ESBMC_get_unique_address (loose default,
// solidity_address.c:84-107) unrolls __ESBMC_assume(tmp != sol_addr_array[i])
// for i in 0..15. The 16th-prior-allocation slot (sol_addr_array[16]) is
// NOT covered by any if in the chain, so the 18th allocation onward is
// unconstrained against slot 16. SMT can pick the 18th allocation's
// $address equal to the 17th's, violating pairwise distinctness.
//
// Fix S1.2: switch the default routing to the precise for-loop variant
// in solidity_address.c.
contract Box {}

contract H {
    Box[18] vs;
    constructor() {
        for (uint i = 0; i < 18; i++) vs[i] = new Box();
    }
    function check_pairwise_distinct_17_18() public view {
        // The 17th and 18th allocations (indices 16 and 17) — the 18th
        // is the first allocation with no constraint against the 17th's
        // $address (the unrolled chain ends at slot 15).
        assert(address(vs[16]) != address(vs[17]));
    }
}
