/* Test: assigns compliance with pointer dereference — FAIL
 * Function foo declares assigns(*ptr) but also directly writes h.
 * Since ptr == &g (from caller), ptr != &h, so the aliasing disjunction
 * for h is false and the compliance check should fail.
 */

int g = 0;
int h = 0;

void foo(int *ptr)
{
    __ESBMC_assigns(*ptr);
    __ESBMC_ensures(*ptr == 42);

    *ptr = 42;
    h = 99;  // VIOLATION: h not in assigns clause, ptr != &h
}

int main()
{
    foo(&g);
    return 0;
}
