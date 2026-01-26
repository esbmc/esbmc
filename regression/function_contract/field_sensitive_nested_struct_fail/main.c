/* =============================================================================
 * TEST: Field-Sensitive Nested Struct — FAIL case
 * =============================================================================
 *
 * PURPOSE:
 *   Verify that the field-sensitive havoc correctly randomizes the target
 *   field. When the ensures clause does NOT constrain the havoc'd field,
 *   asserting a specific value for it must FAIL.
 *
 * WHAT THIS TESTS:
 *   - assigns(p->inner.field_a) havocs field_a
 *   - The ensures clause says nothing about field_a's final value
 *   - Therefore, asserting field_a == 99 after replacement should FAIL
 *   - But field_b should still be preserved (not in assigns)
 *
 * EXPECTED: VERIFICATION FAILED
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

/* Contract: modifies field_a but does NOT constrain its final value */
void update_nested_unconstrained(Outer *p, int new_val) {
    __ESBMC_requires(p != NULL);
    __ESBMC_assigns(p->inner.field_a);
    /* No ensures about field_a — it is left unconstrained after havoc */
    __ESBMC_ensures(p->inner.field_b == __ESBMC_old(p->inner.field_b));
    __ESBMC_ensures(p->outer_val == __ESBMC_old(p->outer_val));

    p->inner.field_a = new_val;
}

int main() {
    Outer obj;
    obj.inner.field_a = 10;
    obj.inner.field_b = 20;
    obj.outer_val = 30;

    update_nested_unconstrained(&obj, 99);

    /* field_b and outer_val are preserved */
    assert(obj.inner.field_b == 20);  /* PASS — not in assigns */
    assert(obj.outer_val == 30);      /* PASS — not in assigns */

    /* field_a was havoc'd and not constrained — this MUST fail */
    assert(obj.inner.field_a == 99);  /* FAIL — unconstrained havoc */

    return 0;
}
