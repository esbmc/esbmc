/* Minimal test case for goto_program crash with multiple __ESBMC_old
 * 
 * Bug: When --enforce-contract is used with ensures containing multiple
 * __ESBMC_old() calls, ESBMC crashes with:
 *   Assertion `t->target_number != 0' failed
 *   in goto_program.cpp:378
 * 
 * This happens during compute_target_numbers() when contract processing
 * generates GOTO instructions with targets that are not properly marked.
 */
#include <stddef.h>

typedef struct {
    int x;
    int y;
} Point;

void modify_point(Point *p, int idx)
{
    __ESBMC_requires(p != NULL);
    __ESBMC_requires(idx >= 0 && idx < 2);
    
    __ESBMC_assigns(p[0].x, p[0].y, p[1].x, p[1].y);
    
    // Frame conditions with many __ESBMC_old
    __ESBMC_ensures(idx != 0 || (p[0].x == 10 && p[1].x == __ESBMC_old(p[1].x) && p[1].y == __ESBMC_old(p[1].y)));
    __ESBMC_ensures(idx != 1 || (p[1].x == 10 && p[0].x == __ESBMC_old(p[0].x) && p[0].y == __ESBMC_old(p[0].y)));
    
    p[idx].x = 10;
}

int main()
{
    Point pts[2];
    modify_point(pts, 0);
    return 0;
}
