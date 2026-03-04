/* Test: Contract annotation precondition violation
 * 
 * This test verifies that precondition violation is detected in replace mode.
 * The replace mode checks preconditions at call site.
 */

__ESBMC_contract
void require_positive(int x)
{
    __ESBMC_requires(x > 0);
    // x must be positive
}

int main(void)
{
    require_positive(-5);  // Violation: -5 is not > 0
    return 0;
}
