/* Test: assigns compliance FAIL
 * Function foo modifies 'h' which is NOT declared in __ESBMC_assigns.
 * This is a violation of the assigns clause.
 */

int g = 0;
int h = 0;

void foo(void)
{
    __ESBMC_assigns(g);
    __ESBMC_ensures(g == 42);

    g = 42;
    h = 99;  // VIOLATION: h not in assigns clause
}
