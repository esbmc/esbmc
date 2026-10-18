// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// `delete bytesVar` followed by `b.push(...)` must produce a freshly
// initialised buffer that supports element appends. Originally
// hypothesized as Bug #2 (init flag), refuted empirically — the test
// passes today.
//
// CORE: locks current correctness. The upcoming emit_delete_block
// refactor will explicitly write `initialized=1` so future divergence
// (e.g. an SMT model that does enforce init checks more strictly) won't
// regress this case.
contract C {
    bytes b;

    function f() public {
        require(b.length == 0);
        b.push(0x41);
        b.push(0x42);
        delete b;
        b.push(0x43);
        assert(b.length == 1);
        assert(b[0] == 0x43);
    }
}
