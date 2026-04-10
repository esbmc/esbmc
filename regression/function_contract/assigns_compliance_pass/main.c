/* Test: assigns compliance PASS
 * Function foo only modifies 'g' which is declared in __ESBMC_assigns.
 * Global 'h' is not modified, so compliance should pass.
 */

int g = 0;
int h = 0;

void foo(void)
{
    __ESBMC_assigns(g);
    __ESBMC_ensures(g == 42);

    g = 42;
    // h is NOT modified — compliance should pass
}
