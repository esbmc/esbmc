/* memleak_check_enforce_pass: --enforce-contract + --memory-leak-check must
 * compose. The wrapper allocates a buffer for the pointer parameter so the
 * body can dereference it; that allocation is wrapper-internal and must be
 * freed before the wrapper returns. Without the matching free the leak
 * checker reports CWE-401 against the user's function (issue #4908).
 */
__ESBMC_contract
int read_thru(const int *p)
{
    __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(*p)));
    __ESBMC_ensures(__ESBMC_return_value == *p);
    __ESBMC_assigns();
    return *p;
}

int main(void) { return 0; }
