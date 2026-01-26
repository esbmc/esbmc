/* =============================================================================
 * TEST: Field-Sensitive Nested Struct — Pillar 1 (Access Path Restoration)
 * =============================================================================
 * 
 * PURPOSE:
 *   Demonstrate that ESBMC's field-sensitive assigns clause can track
 *   modifications to individual fields within nested structures accessed
 *   through pointers. This is a key differentiator vs. CBMC's field-insensitive
 *   approach, which would havoc the entire struct on any member write.
 *
 * WHAT THIS TESTS:
 *   - assigns(p->inner.field_a) only havocs that specific nested field
 *   - p->inner.field_b and p->outer_val remain unchanged
 *   - The access path "p -> inner -> field_a" is correctly restored from
 *     the underlying byte offset
 *
 * CBMC COMPARISON:
 *   In CBMC's frame condition analysis, writing to any member of a struct
 *   pointer target typically havocs the entire pointed-to object. Our system
 *   tracks the precise member path, enabling minimal havoc.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>

typedef struct {
    int field_a;
    int field_b;
} Inner;

typedef struct {
    Inner inner;
    int outer_val;
} Outer;

/* Contract: only modifies the nested field p->inner.field_a */
void update_nested_field(Outer *p, int new_val) {
    __ESBMC_requires(p != NULL);
    __ESBMC_assigns(p->inner.field_a);
    __ESBMC_ensures(p->inner.field_a == new_val);
    __ESBMC_ensures(p->inner.field_b == __ESBMC_old(p->inner.field_b));
    __ESBMC_ensures(p->outer_val == __ESBMC_old(p->outer_val));

    p->inner.field_a = new_val;
}

int main() {
    Outer obj;
    obj.inner.field_a = 10;
    obj.inner.field_b = 20;
    obj.outer_val = 30;

    update_nested_field(&obj, 99);

    /* Field-sensitive verification:
     * Only field_a was in the assigns clause, so field_b and outer_val
     * must retain their original values after replacement. */
    assert(obj.inner.field_a == 99);  /* Modified — contract guarantees */
    assert(obj.inner.field_b == 20);  /* Unchanged — not in assigns */
    assert(obj.outer_val == 30);      /* Unchanged — not in assigns */

    return 0;
}
