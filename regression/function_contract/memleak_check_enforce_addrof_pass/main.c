/* memleak_check_enforce_addrof_pass: covers the address-of surface form of
 * __ESBMC_is_fresh combined with --memory-leak-check. The bare-pointer form
 * is covered by memleak_check_enforce_pass; this case exercises the
 * address-of peel branch (stripped = &p -> ptr_var = p) at the same time as
 * the wrapper-free fix for issue #4908.
 */
__ESBMC_contract
int read_thru_addrof(const int *p)
{
    __ESBMC_requires(__ESBMC_is_fresh(&p, sizeof(*p)));
    __ESBMC_ensures(__ESBMC_return_value == *p);
    __ESBMC_assigns();
    return *p;
}

int main(void) { return 0; }
