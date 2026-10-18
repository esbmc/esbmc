/* Stage 0.5' KNOWNBUG: pinpoints the value-set / points-to gap that
 * causes 7 of the napp_* nested-array stress tests to verdict-mismatch
 * post-Stage-1a (commit 8eee9ac09d).
 *
 * Bug shape (general ESBMC, NOT Solidity-specific):
 *   - Outer pointer-array allocated on the heap (here: malloc).
 *   - Successive `flags[i] = ptr` writes through the heap pointer.
 *   - A later read `arr = flags[i]` is encoded as a CONCAT of byte-
 *     extracts on the underlying allocation buffer.
 *   - The assertion encoding uses `SAME-OBJECT(arr, &candidate)` which
 *     consults value-set / points-to.
 *   - Value-set tracks TYPED-pointer assignments. It does NOT scan
 *     byte-level WITH chains for a single-pointer source. So the read
 *     side's points-to set is empty, the SAME-OBJECT chain falls
 *     through to `invalid_object`, and the SMT model picks a value
 *     that satisfies the negated assertion.
 *
 * SSA evidence (excerpt from --ssa-trace on this file):
 *   write: dynamic_1_array#k+1 == (dynamic_1_array#k WITH [8 := byte0])
 *          ... (eight WITHs at indices 8..15, one byte of the pointer each)
 *   read:  arr == (void*)(CONCAT(... eight byte-extracts of dynamic_1_array[8..15]))
 *   assertion: SAME-OBJECT(arr, &realloc_4_array) ? ...
 *              SAME-OBJECT(arr, &dynamic_2_array) ? ...
 *              ... : invalid_object   ← model picks here
 *
 * The companion test value_set_byte_with_stack_pass uses
 *   void *flags[3];
 * (stack/global), where ESBMC encodes the writes as a single typed
 * WITH on a typed `void*[3]` SMT entity, value-set tracks the pointer
 * assignment, and the same property holds. That control passes.
 *
 * Stage 1 fix surface (NOT applied):
 *   (a) src/pointer-analysis/value_set.cpp read path — when the read
 *       expression resolves to a CONCAT of byte-extracts on a heap
 *       buffer with a coherent WITH-chain emitting the bytes of a
 *       single typed pointer source, propagate that source's points-to.
 *   (b) frontend/symex lowering — emit a single typed-element WITH
 *       through a typed view of the heap buffer for `T**`-typed
 *       indexed writes, instead of byte-decomposed writes.
 *
 * Diagnosis log: notes/napp/stage0_5p/CONCLUSION.md.
 *
 * NOT a fix target:
 *   - solidity_array.c — the helper is a victim, not the cause.
 *   - symex_function.cpp parameter binding — works correctly (see R2/R7b).
 *   - _ESBMC_array_push element preservation — wrong hypothesis,
 *     Stage 1b reverted.
 */
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void *header_push(void *array, void *element, size_t elem_size) {
    if (array == NULL) {
        size_t *block = (size_t *)malloc(sizeof(size_t) + elem_size);
        __ESBMC_assume(block != 0);
        block[0] = 1;
        void *data = (void *)(block + 1);
        memcpy(data, element, elem_size);
        return data;
    }
    size_t old_len = ((size_t *)array)[-1];
    size_t new_len = old_len + 1;
    size_t *old_block = ((size_t *)array) - 1;
    size_t *new_block = (size_t *)realloc(
        old_block, sizeof(size_t) + new_len * elem_size);
    __ESBMC_assume(new_block != 0);
    new_block[0] = new_len;
    void *new_data = (void *)(new_block + 1);
    memcpy((char *)new_data + old_len * elem_size, element, elem_size);
    return new_data;
}

int main(void) {
    /* HEAP-allocated outer pointer-array — the trigger.
     * Replace this with `void *flags[3] = {0};` to flip to PASS. */
    void **flags = (void **)malloc(3 * sizeof(void *));
    __ESBMC_assume(flags != 0);
    flags[0] = 0;
    flags[1] = 0;
    flags[2] = 0;

    _Bool v0 = 0, v1 = 1;
    flags[1] = header_push(flags[1], &v0, sizeof(_Bool));
    flags[1] = header_push(flags[1], &v1, sizeof(_Bool));

    _Bool *arr = (_Bool *)flags[1];
    assert(arr != NULL);
    assert(arr[0] == 0);
    assert(arr[1] == 1);
    return 0;
}
