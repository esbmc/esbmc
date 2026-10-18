/* Stage 0.5' control case (companion to value_set_byte_with_heap_knownbug).
 *
 * Successive index writes through a STACK-allocated outer pointer-array
 * propagate correctly to subsequent reads. ESBMC encodes
 *   void *flags[3];
 * as a typed `void *[3]` SMT entity; `flags[1] = ptr` is a single
 * typed-element WITH that updates value-set's points-to set. Reads of
 * `flags[1]` consult points-to and resolve to the right object.
 *
 * The HEAP variant of this same logic fails — see the
 * value_set_byte_with_heap_knownbug companion for the bug shape and
 * the SSA-level explanation.
 *
 * Verifies under the default Stage-1a soundness contract for the
 * generic dyn-array push helper (commit 8eee9ac09d) — the realloc
 * non-null assume that closes a separate, independent leak.
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
    void *flags[3] = {0};   /* STACK fixed-size array — typed void*[3] in SMT */

    _Bool v0 = 0, v1 = 1;
    flags[1] = header_push(flags[1], &v0, sizeof(_Bool));
    flags[1] = header_push(flags[1], &v1, sizeof(_Bool));

    _Bool *arr = (_Bool *)flags[1];
    assert(arr != NULL);
    assert(arr[0] == 0);
    assert(arr[1] == 1);
    return 0;
}
