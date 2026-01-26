/* =============================================================================
 * TEST: Deep Nesting (3 levels) — Pillar 1 (Access Path Restoration)
 * =============================================================================
 *
 * PURPOSE:
 *   Test that the access path restoration algorithm can handle 3-level deep
 *   struct nesting: p->level1.level2.target_field. This exercises the
 *   recursive offset-to-member mapping:
 *     byte_offset -> level1(offset) -> level2(sub-offset) -> target_field
 *
 * SIGNIFICANCE FOR PAPER:
 *   Real embedded systems (e.g., automotive ECUs) frequently use deeply nested
 *   configuration structs. If the verifier cannot track field-level modifications
 *   through deep nesting, it must conservatively havoc entire sub-trees.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>

typedef struct {
    int target_field;
    int sibling_field;
} Level2;

typedef struct {
    Level2 level2;
    int mid_field;
} Level1;

typedef struct {
    Level1 level1;
    int top_field;
} TopLevel;

/* Only modifies the deepest nested field */
void update_deep(TopLevel *p, int val) {
    __ESBMC_requires(p != NULL);
    __ESBMC_assigns(p->level1.level2.target_field);
    __ESBMC_ensures(p->level1.level2.target_field == val);
    /* All sibling fields at every level must be preserved */
    __ESBMC_ensures(p->level1.level2.sibling_field ==
                    __ESBMC_old(p->level1.level2.sibling_field));
    __ESBMC_ensures(p->level1.mid_field == __ESBMC_old(p->level1.mid_field));
    __ESBMC_ensures(p->top_field == __ESBMC_old(p->top_field));

    p->level1.level2.target_field = val;
}

int main() {
    TopLevel obj;
    obj.level1.level2.target_field = 1;
    obj.level1.level2.sibling_field = 2;
    obj.level1.mid_field = 3;
    obj.top_field = 4;

    update_deep(&obj, 100);

    assert(obj.level1.level2.target_field == 100);  /* Modified */
    assert(obj.level1.level2.sibling_field == 2);    /* Preserved */
    assert(obj.level1.mid_field == 3);               /* Preserved */
    assert(obj.top_field == 4);                      /* Preserved */

    return 0;
}
