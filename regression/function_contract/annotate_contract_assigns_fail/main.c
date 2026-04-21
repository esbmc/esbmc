/* Test: --enforce-all-contracts checks assigns compliance
 * foo declares __ESBMC_assigns(g) but also writes to h.
 * --enforce-all-contracts must catch this violation.
 */

int g = 0;
int h = 0;

__ESBMC_contract
void foo(void)
{
    __ESBMC_assigns(g);
    __ESBMC_ensures(g == 1);

    g = 1;
    h = 99; /* VIOLATION: h not in assigns clause */
}
