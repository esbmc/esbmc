/*
 * Test: 3-level hierarchy with replace-all
 * Expected: VERIFICATION SUCCESSFUL
 *
 * Scenario:
 *   - bottom(x): returns x + 10, contract ensures ret == x + 10
 *   - middle(x): calls bottom(x), adds 5, contract ensures ret == x + 15
 *   - top(x): calls middle(x), adds 1, contract ensures ret == x + 16
 *   - main(): calls top(3), asserts result == 19
 *
 * With --replace-call-with-contract "*":
 *   - bottom is a leaf → replaced with contract (havoc + assume ret == x + 10)
 *   - middle is a parent calling bottom → body KEPT, bottom call replaced
 *   - top is a parent calling middle → body KEPT, middle call replaced? Or kept?
 *
 * This tests whether the hierarchical replacement strategy works
 * correctly for 3+ level call chains with value (non-pointer) returns.
 */
#include <assert.h>

int bottom(int x)
{
    __ESBMC_requires(x >= 0 && x <= 100);
    __ESBMC_ensures(__ESBMC_return_value == x + 10);

    return x + 10;
}

int middle(int x)
{
    __ESBMC_requires(x >= 0 && x <= 100);
    __ESBMC_ensures(__ESBMC_return_value == x + 15);

    int b = bottom(x);
    return b + 5;
}

int top(int x)
{
    __ESBMC_requires(x >= 0 && x <= 100);
    __ESBMC_ensures(__ESBMC_return_value == x + 16);

    int m = middle(x);
    return m + 1;
}

int main(void)
{
    int r = top(3);
    assert(r == 19);
    return 0;
}
