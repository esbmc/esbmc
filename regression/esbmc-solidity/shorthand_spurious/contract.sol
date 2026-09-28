// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// `shx` reaches the array-size expression twice, once through `shy`, so the
// printer sees two exprts for one symbol that differ in something other than the
// identifier. That is not a namespace collision, and the dump must keep `shx`
// short rather than fall back to the mangled id. Names are deliberately
// distinctive so the expected regexes cannot be matched by an unrelated symbol.
uint constant shx = 1;
uint constant shy = shx;

contract Base {
    int[shy] arr;
    constructor() {
        arr[0] = 0;
    }
}
