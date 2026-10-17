// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

// Stage 3 mapping refactor exhibitor: many writes followed by many reads
// on the SAME mapping, all under k-induction.  Pre-Stage-2 the linked-list
// `_ESBMC_Mapping` walked O(N) per access, so the SMT formula at k=20 grew
// quadratically and timed out.  Pre-Stage-3 the per-access still cast
// `(uintptr_t)m->base` to derive the per-state-var ID, generating a
// pointer-to-uint64 SMT extraction that the solver had to discharge for
// every select/store.  Stage 3 reads the explicit `m->mid` field directly,
// turning each access into one bare bv64 select/store on the global
// `_ESBMC_map_storage` array.
//
// Structure: 8 distinct keys written then read back in a single function.
// Properties hold trivially under single-call semantics; this test exists
// to verify the SMT-array path closes within the regression timeout.
contract MapSmtArrayWriteRead {
    mapping(uint256 => uint256) public m;

    function check() external {
        m[0] = 100;
        m[1] = 101;
        m[2] = 102;
        m[3] = 103;
        m[4] = 104;
        m[5] = 105;
        m[6] = 106;
        m[7] = 107;

        assert(m[0] == 100);
        assert(m[1] == 101);
        assert(m[2] == 102);
        assert(m[3] == 103);
        assert(m[4] == 104);
        assert(m[5] == 105);
        assert(m[6] == 106);
        assert(m[7] == 107);
    }
}
