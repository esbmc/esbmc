/* memleak_check_enforce_fail: --memory-leak-check must still catch real
 * user-introduced leaks inside a contract-enforced function. Companion to
 * memleak_check_enforce_pass: ensures the wrapper-free fix for issue #4908
 * does not over-suppress and hide genuine CWE-401 violations.
 *
 * The body mallocs a buffer and never frees it; the only auto-allocation the
 * wrapper does (for p) is freed by the fix, so the leaked buffer reported by
 * the checker must be the body's malloc — confirming the checker is alive.
 */
#include <stdlib.h>

__ESBMC_contract
int read_thru(const int *p)
{
    __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(*p)));
    __ESBMC_assigns();

    int *leaked = (int *)malloc(sizeof(int));
    if (leaked)
        *leaked = *p;
    return *p;
}

int main(void) { return 0; }
