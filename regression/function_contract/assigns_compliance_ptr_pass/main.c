/* Test: assigns compliance with pointer dereference — PASS
 * Function foo declares assigns(*ptr) and only writes through ptr.
 * When called with ptr == &g, the aliasing disjunction allows g to change.
 * Global h is not modified, so compliance should pass.
 */

int g = 0;
int h = 0;

void foo(int *ptr)
{
    __ESBMC_assigns(*ptr);
    __ESBMC_ensures(*ptr == 42);

    *ptr = 42;
    // h is NOT modified — compliance should pass
}

int main()
{
    foo(&g);
    return 0;
}
