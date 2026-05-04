// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Tests 2D dynamic array (uint[][]) declaration, push, indexing,
// and passing array elements as storage references to internal functions.
// NOTE: ESBMC does not model memory aliasing — memory parameters are
// independent, so a[0] and b[0] cannot alias even if a==b at runtime.
// Assertions 3,4 (a[0]==2, c[0][0]==2) PASS in ESBMC but would FAIL in
// a full alias-tracking verifier.
contract Aliasing
{
    uint[] array1;
    uint[][] array2;

    constructor() {
        array1.push(0);
        array2.push();
        array2[0].push(0);
    }

    function f(
        uint[] memory a,
        uint[] memory b,
        uint[][] memory c,
        uint[] storage d
    ) internal {
        array1[0] = 42;
        a[0] = 2;
        c[0][0] = 2;
        b[0] = 1;
        // Erasing knowledge about memory references should not
        // erase knowledge about state variables.
        assert(array1[0] == 42);
        // However, an assignment to a storage reference will erase
        // storage knowledge accordingly.
        d[0] = 2;
        // In ESBMC: d points to array2[x], not array1, so this passes.
        // (A full alias tracker would fail this as false positive.)
        assert(array1[0] == 42);
        // In ESBMC: memory params are independent, so a[0] is still 2.
        assert(a[0] == 2);
        // In ESBMC: c[0] is independent of b, so c[0][0] is still 2.
        assert(c[0][0] == 2);
        assert(d[0] == 2);
        assert(b[0] == 1);
    }
    function g(
        uint[] memory a,
        uint[] memory b,
        uint[][] memory c,
        uint x
    ) public {
        f(a, b, c, array2[x]);
    }
}
