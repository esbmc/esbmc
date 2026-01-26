/* =============================================================================
 * TEST: Auto Assigns Inference (Nested Struct) — Pillar 2
 * =============================================================================
 *
 * PURPOSE:
 *   Verify that auto-inference correctly identifies modifications to
 *   specific fields of a struct passed by pointer. The function modifies
 *   p->x and p->y but NOT p->z. Auto-inference should produce:
 *     assigns(p->x, p->y)
 *   instead of the conservative:
 *     assigns(*p)  // which would havoc z too
 *
 * PAPER VALUE:
 *   This directly demonstrates the "minimal havoc" advantage. If auto-inference
 *   is field-sensitive, it eliminates the need for manual assigns annotations
 *   in the vast majority of industrial code.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>

typedef struct {
    int x;
    int y;
    int z;
} Vec3;

/* Modifies only x and y; z is read-only */
void translate_xy(Vec3 *v, int dx, int dy) {
    __ESBMC_requires(v != NULL);
    __ESBMC_assigns(v->x, v->y);
    __ESBMC_ensures(v->x == __ESBMC_old(v->x) + dx);
    __ESBMC_ensures(v->y == __ESBMC_old(v->y) + dy);
    __ESBMC_ensures(v->z == __ESBMC_old(v->z));

    v->x += dx;
    v->y += dy;
    /* v->z is NOT modified */
}

int main() {
    Vec3 pos;
    pos.x = 10;
    pos.y = 20;
    pos.z = 30;

    translate_xy(&pos, 5, -3);

    assert(pos.x == 15);
    assert(pos.y == 17);
    assert(pos.z == 30);  /* Must remain unchanged */

    return 0;
}
