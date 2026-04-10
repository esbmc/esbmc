/* Test that --enforce-contract works with __ESBMC_assigns
 * 
 * This test verifies the fix for the crash that occurred when
 * using --enforce-contract mode with functions containing
 * __ESBMC_assigns clauses.
 * 
 * Previously, ESBMC would crash with:
 *   Assertion `0 && "unexpected side effect"' failed
 *   in symex_assign.cpp:251
 * 
 * The fix adds a case for sideeffect2t::assigns_target in
 * handle_sideeffect() to properly handle assigns during symex.
 */
#include <assert.h>
#include <stddef.h>

typedef struct {
    int x;
    int y;
    int z;
} Point;

void init_point(Point *p)
{
    __ESBMC_requires(p != NULL);
    
    __ESBMC_assigns(p->x, p->y, p->z);
    __ESBMC_ensures(p->x == 0);
    __ESBMC_ensures(p->y == 0);
    __ESBMC_ensures(p->z == 0);
    
    p->x = 0;
    p->y = 0;
    p->z = 0;
}

int main()
{
    Point p;
    p.x = 10;
    p.y = 20;
    p.z = 30;
    
    init_point(&p);
    
    assert(p.x == 0);
    assert(p.y == 0);
    assert(p.z == 0);
    
    return 0;
}
