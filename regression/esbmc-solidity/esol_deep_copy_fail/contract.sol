// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract Counter {
    uint public x;

    constructor() {
        x = 3;
    }

    function add(uint n) public {
        x = x + n;
    }

    function mul(uint n) public {
        x = x * n;
    }
}

// User-written TOD harness demonstrating the __ESOL_deep_copy
// intrinsic.  The stub below lets solc accept the call syntax; the
// ESBMC Solidity frontend intercepts every call to
// `__ESOL_deep_copy` and rewrites it to `_ESBMC_clone_<C>(arg)`.
function __ESOL_deep_copy(Counter src) pure returns (Counter) {
    return src;
}

contract Harness {
    function check(uint n1, uint n2) public {
        Counter a = new Counter();
        Counter b = __ESOL_deep_copy(a);

        a.add(n1); a.mul(n2);
        b.mul(n2); b.add(n1);

        assert(a.x() == b.x());
    }
}
