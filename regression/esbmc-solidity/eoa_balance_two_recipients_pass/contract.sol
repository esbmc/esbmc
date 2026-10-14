// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Two distinct EOAs receive distinct amounts; their balances must be
// independently tracked.  Exercises sol_eoa_addr_array's per-slot
// indexing (no aliasing between distinct addresses).
contract Bal {
    constructor() payable {}

    function check(address payable a, address payable b, uint x, uint y) public {
        if (a == b) return;
        if (a == address(this) || b == address(this)) return;
        // Pin both starting balances to 0.
        if (a.balance != 0 || b.balance != 0) return;
        if (address(this).balance < x) return;
        a.transfer(x);
        if (address(this).balance < y) return;
        b.transfer(y);
        assert(a.balance == x);
        assert(b.balance == y);
    }
}
// KNOWNBUG (performance): CVC5 solved this in ~52s when the test was added;
// the frontend's encoding has since grown and the solve now takes ~280s, past
// every CI timeout tier. The verdict is still VERIFICATION SUCCESSFUL.
