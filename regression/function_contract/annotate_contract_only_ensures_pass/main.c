/* Test: Contract annotation with only ensures (no requires, no assigns)
 * 
 * This test verifies that a function with only postcondition works correctly.
 * requires defaults to true, assigns defaults to conservative havoc.
 */

int result = 0;

__ESBMC_contract
int get_positive(void)
{
    __ESBMC_ensures(__ESBMC_return_value > 0);
    // No requires, no assigns
    // requires defaults to true
    // assigns defaults to conservative havoc
    
    result = 42;
    return result;
}

int main(void)
{
    int x = get_positive();
    return 0;
}
