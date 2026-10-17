/* Pins an ESBMC C++-frontend bug isolated 2026-05-10. Reproduces the
 * cluster A class of failures in regression/esbmc-solidity/napp_*.
 *
 * KEY: this exact source compiled as C (.c extension) PASSES under the
 * same flags. As C++ (.cpp extension) it FAILS. Bug is in the C++
 * frontend's lowering, NOT in symex_realloc nor pointer-analysis
 * proper (those work on the C-lowered IR).
 *
 * Bug shape (precise trigger, see notes/napp/phase12_pure_c_repro/):
 *   1. helper function takes pointer parameter `T* p`
 *   2. helper INTERNALLY reads from `*p` (any way: typed cast, struct
 *      member access, byte-CONCAT load) BEFORE the realloc call
 *   3. helper calls `realloc(p_or_derived, ...)`
 *   4. caller chains 3+ explicit-unrolled invocations of the helper
 *      (no for-loop)
 *
 * Workarounds (all verified PASSING under same C++ flags):
 *   - rename to .c (use C frontend): PASSES
 *   - read `*p` AFTER realloc instead of before
 *   - lift the read to caller side and pass length as helper parameter
 *   - replace explicit-unrolled calls with `for (i; i<N; i++)` loop
 *
 * The Solidity library helper `_ESBMC_array_push` in
 * src/c2goto/library/solidity/solidity_array.c is COMPILED AS C in
 * the c2goto pipeline, so the C frontend handles it. But the SAME
 * library is invoked from Solidity-frontend-emitted code that goes
 * through the C++ frontend's lowering pipeline (Solidity contracts
 * use class semantics → cpp-frontend), and THAT path triggers the bug.
 *
 * EXACT bug surface (narrowed via 2026-05-10 phase 12 investigation):
 *   Run with --show-symex-value-sets and grep for the `a` parameter:
 *
 *   C version (PASS):
 *     @push_int@a?2 = { NULL, <realloc_2_array, 8, 16, uchar [4*new_len?1#1+8]> }
 *
 *   C++ version (FAIL):
 *     @push_int@a::0?2 = { NULL, <realloc_2_array, 8, 16, uchar [4*new_len?1#0+8]> }
 *
 *   ONE digit difference: `#1` (post-assignment, deterministic = old_len+1)
 *   vs `#0` (pre-assignment SSA version, unconstrained nondet).
 *
 *   The C++ frontend's value-set capture of the candidate's size expression
 *   uses the WRONG SSA version of `new_len`. Because `new_len#0` is
 *   unconstrained, SAT can pick old_size != 12, making the realloc
 *   preservation chain's `idx < min(old_size, new_size)` bound check
 *   nondet → preserved bytes outside the (narrowed) bound stay
 *   uninit-fallback → assertion fails.
 *
 * Likely fix loci:
 *   - src/pointer-analysis/value_set.cpp — when capturing object_descriptor
 *     for derived pointers, ensure size expression uses RENAMED
 *     (post-assignment) SSA versions, not pre-rename versions
 *   - src/goto-symex/symex_function.cpp — function-call return value
 *     transfer may not propagate cur_state->rename through size exprs
 *
 * Diagnostics: notes/napp/phase12_pure_c_repro/CONCLUSION.md
 * When this bug is fixed, flip mode to CORE and the test should pass.
 */

#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

static inline size_t _hdr_read(void* a) { return ((size_t*)a)[-1]; }

static void* push_int(void* a, int v) {
    /* This typed-load-through-cast triggers the bug when paired with
     * the realloc below across 3+ unrolled calls. */
    size_t old_len = (a == 0) ? 0 : _hdr_read(a);
    size_t new_len = old_len + 1;
    size_t* old_block = (a == 0) ? 0 : ((size_t*)a) - 1;
    size_t* nb = (size_t*)realloc((void*)old_block,
                                   sizeof(size_t) + new_len * sizeof(int));
    __ESBMC_assume(nb != 0);
    nb[0] = new_len;
    int* d = (int*)(nb + 1);
    d[old_len] = v;
    return (void*)d;
}

int main() {
    int* arr = 0;
    arr = (int*)push_int((void*)arr, 100);
    arr = (int*)push_int((void*)arr, 200);
    arr = (int*)push_int((void*)arr, 300);
    /* arr[1] should be 200 (preserved across realloc chain) but
     * SAT can pick a model where preserved bytes are unconstrained.
     * Verifies SUCCESSFUL after the symex/value-set bug is fixed. */
    assert(arr[1] == 200);
    return 0;
}
