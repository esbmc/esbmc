/* C-suite KNOWNBUG: heap byte-WITH points-to gap (frontend-agnostic).
 *
 * This is a C-language pin of a value-set / SAME-OBJECT bug originally
 * surfaced via the napp_* nested-array stress matrix (commit 43ba13446c)
 * and first pinned in the Solidity test suite at
 *   regression/esbmc-solidity/value_set_byte_with_heap_knownbug/bug.c
 * (commit 1e3bd6e134). The source is plain C with no Solidity intrinsics;
 * placing it in the default C suite proves the gap is C-level — visible
 * to anyone running the standard regression set, not just Solidity work.
 *
 * Bug shape:
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
 *              ... : invalid_object   <- model picks here
 *
 * The companion test pointer_provenance_byte_stack_pass uses
 *   void *flags[3];
 * (stack), where ESBMC encodes the writes as a single typed WITH on a
 * typed `void*[3]` SMT entity, value-set tracks the pointer assignment,
 * and the same property holds. That control passes.
 *
 * Four falsified fix surfaces (Stage-0 probes documented under
 * notes/napp/stage0_pathA, stage0_pathb, stage0_h1, stage0_h2):
 *   path (a) read-side scan in value_set.cpp        -- keys match; downstream
 *   Option B at H1 in dereference::construct_from_array -- WITH not visible
 *   path (b) write-side typed-WITH in symex_assign_concat -- can't reject risk3/6
 *   H2 SSA-inlining + concat2t simplifier repair    -- if2t merge in chain
 *
 * KNOWNBUG until an architectural fix surface (see
 * notes/napp/heap_byte_provenance/fix_directions.md) lands.
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
    /* HEAP-allocated outer pointer-array -- the trigger.
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
