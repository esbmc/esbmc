/* Test: Contract annotation replace mode precondition violation
 * 
 * In replace mode, the precondition is checked at the call site.
 */

__attribute__((annotate("__ESBMC_contract")))
int double_positive(int x)
{
    __ESBMC_requires(x > 0);
    __ESBMC_ensures(__ESBMC_return_value == x * 2);
    
    return x * 2;
}

int main(void)
{
    int result = double_positive(0);  // Violation: 0 is not > 0
    return 0;
}
