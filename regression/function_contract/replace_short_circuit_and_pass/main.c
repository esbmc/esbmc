// Test: ensures clause with nested && chain and __ESBMC_old on struct array.
// Validates that inline_temporary_variables correctly reconstructs
// short-circuit AND chains (a && b && c && d) from Clang's cascading control flow.
#include <assert.h>

typedef struct {
    int x;
    int y;
    int z;
} Point;

void modify_point(Point* p, int which)
{
    __ESBMC_requires(p != 0);
    __ESBMC_requires(which >= 0 && which <= 2);

    __ESBMC_assigns(p->x, p->y, p->z);

    // If which != 0, all three fields preserved (nested && chain)
    __ESBMC_ensures(which != 0 || (
        p->y == __ESBMC_old(p->y) &&
        p->z == __ESBMC_old(p->z)
    ));

    // If which != 1, x and z preserved
    __ESBMC_ensures(which != 1 || (
        p->x == __ESBMC_old(p->x) &&
        p->z == __ESBMC_old(p->z)
    ));

    // If which != 2, x and y preserved
    __ESBMC_ensures(which != 2 || (
        p->x == __ESBMC_old(p->x) &&
        p->y == __ESBMC_old(p->y)
    ));

    if (which == 0) {
        p->x = 100;
    } else if (which == 1) {
        p->y = 200;
    } else {
        p->z = 300;
    }
}

int main()
{
    Point pts[3];
    pts[0].x = 1; pts[0].y = 2; pts[0].z = 3;

    // Only modify x (which==0), y and z should stay
    modify_point(&pts[0], 0);

    assert(pts[0].y == 2);
    assert(pts[0].z == 3);

    return 0;
}
