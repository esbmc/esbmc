/* Test: assigns compliance with multiple pointer dereferences — PASS
 * Function foo declares assigns(*p, *q) and writes through both pointers.
 * Global z is not modified, so compliance should pass.
 */

int x = 0;
int y = 0;
int z = 0;

void foo(int *p, int *q)
{
    __ESBMC_assigns(*p, *q);
    __ESBMC_ensures(*p == 1);

    *p = 1;
    *q = 2;
    // z is NOT modified — compliance should pass
}

int main()
{
    foo(&x, &y);
    return 0;
}
