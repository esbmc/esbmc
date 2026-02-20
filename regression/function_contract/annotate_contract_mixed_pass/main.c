/* Test: Mix of annotated and non-annotated functions
 *
 * This test verifies that --enforce-all-contracts only processes
 * functions marked with __ESBMC_contract annotation.
 * add() is NOT annotated and should NOT be enforced.
 * accumulate() IS annotated and should be enforced.
 */

int global_sum = 0;

// Regular function WITHOUT annotation - not enforced
int add(int a, int b)
{
    return a + b;
}

// Annotation-marked function - will be enforced
__ESBMC_contract
void accumulate(int x)
{
    __ESBMC_requires(x >= 0);
    __ESBMC_assigns(global_sum);
    __ESBMC_ensures(global_sum == __ESBMC_old(global_sum) + x);

    global_sum = global_sum + x;
}

int main(void)
{
    int result = add(3, 4);
    __ESBMC_assert(result == 7, "3 + 4 should be 7");

    global_sum = 0;
    accumulate(10);
    __ESBMC_assert(global_sum == 10, "global_sum should be 10");

    return 0;
}
