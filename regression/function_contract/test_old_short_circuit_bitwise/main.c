// Test that __ESBMC_and, __ESBMC_or, __ESBMC_implies macros can be used to
// avoid short-circuit evaluation issues when using __ESBMC_old() in ensures.
//
// Problem: C's && and || operators have short-circuit evaluation semantics,
// which causes Clang to generate control flow that splits the ensures
// expression, making it impossible for ESBMC to extract the full contract.
//
// Solution: Use __ESBMC_and, __ESBMC_or, __ESBMC_implies macros which expand
// to bitwise operators that evaluate all operands without short-circuiting.

typedef struct {
    int x;
    int y;
} Point;

void update(Point* p, int val) {
    __ESBMC_requires(p != 0);
    __ESBMC_assigns(p->x, p->y);
    // Using helper macros to avoid short-circuit evaluation:
    // Case 1: val > 0 -> update x, keep y
    // Case 2: val <= 0 -> keep both unchanged
    __ESBMC_ensures(
        __ESBMC_or(
            __ESBMC_and(__ESBMC_and(val > 0, p->x == val), p->y == __ESBMC_old(p->y)),
            __ESBMC_and(__ESBMC_and(val <= 0, p->x == __ESBMC_old(p->x)), p->y == __ESBMC_old(p->y))
        )
    );
    
    if (val > 0) {
        p->x = val;
    }
}

int main() {
    Point pt = {1, 2};
    update(&pt, 5);
    // After update with val=5 > 0: pt.x should be 5, pt.y should still be 2
    assert(pt.x == 5);
    assert(pt.y == 2);
    
    Point pt2 = {10, 20};
    update(&pt2, -3);
    // After update with val=-3 <= 0: pt2 should be unchanged
    assert(pt2.x == 10);
    assert(pt2.y == 20);
    
    return 0;
}
