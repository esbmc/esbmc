/* Test: __ESBMC_assigns with struct field via pointer
 * 
 * Verifies that __ESBMC_assigns(ptr->field) only havocs
 * that specific field, leaving other fields unchanged.
 */
#include <assert.h>
#include <stddef.h>

typedef struct {
    int x;
    int y;
    int z;
} Point;

void set_x(Point *p, int val) {
    __ESBMC_requires(p != NULL);
    __ESBMC_assigns(p->x);
    __ESBMC_ensures(p->x == val);
    __ESBMC_ensures(p->y == __ESBMC_old(p->y));
    __ESBMC_ensures(p->z == __ESBMC_old(p->z));
    
    p->x = val;
}

int main() {
    Point pt;
    pt.x = 1;
    pt.y = 2;
    pt.z = 3;
    
    set_x(&pt, 99);
    
    assert(pt.x == 99);  // Modified
    assert(pt.y == 2);   // Unchanged
    assert(pt.z == 3);   // Unchanged
    
    return 0;
}
