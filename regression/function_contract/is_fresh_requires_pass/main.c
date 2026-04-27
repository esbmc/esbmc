/* is_fresh_requires_pass: __ESBMC_is_fresh in requires (canonical bare-pointer
 * form) must produce a properly-typed allocation so that an in-bounds
 * dereference of the parameter succeeds.
 *
 * Regression for the requires-side codegen: the "else" branch used to emit
 * *p = malloc(...) (deref of uninit p) instead of p = malloc(...).
 */
__ESBMC_contract
int foo(int *p)
{
    __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(int)));
    __ESBMC_ensures(1);

    return *p;
}

int main(void) { return 0; }
