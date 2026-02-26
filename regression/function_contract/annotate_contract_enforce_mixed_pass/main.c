/* Test: enforce-all-contracts with mixed annotated/non-annotated functions
 *
 * add() is NOT annotated -> not enforced, body runs normally.
 * accumulate() IS annotated -> enforced, contract is checked.
 *
 * Only accumulate() should be enforced. add() should not be touched.
 */

int global_sum = 0;

/* Not annotated: not enforced */
int add(int a, int b)
{
    return a + b;
}

/* Annotated: will be enforced */
__attribute__((annotate("__ESBMC_contract")))
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
