/*
 * Test: Contract propagation — enforce parent + replace leaf with pointer output
 * Expected: VERIFICATION SUCCESSFUL
 * Status: KNOWNBUG — enforce+replace with pointer params fails without --assume-nonnull-valid
 *
 * Scenario (modeled after ANSSI X.509 parser):
 *   - leaf(out): writes to *out, ensures *out >= 2 on success (ret == 0)
 *   - parent(eaten): calls leaf(&hdr_len), assigns *eaten = hdr_len
 *     Contract: ensures *eaten > 0 on success
 *
 * Expected behavior with correct contract propagation:
 *   1. Replace leaf() call with: havoc *out; assume(*out >= 2 && ret == 0)
 *   2. Enforce parent() contract: *eaten = hdr_len >= 2 > 0 ✓
 *
 * Bug: When --function parent is used, the enforce wrapper dereferences
 *   *eaten in the ensures assertion, but the harness-generated pointer
 *   doesn't point to valid memory. The ensures reads symex::invalid_object
 *   instead of the actual value written by the original function body.
 *   Adding --assume-nonnull-valid makes it work.
 */
#include <assert.h>

int leaf(int *out)
{
    __ESBMC_requires(out != ((void *)0));
    __ESBMC_assigns(*out);
    __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == -1);
    __ESBMC_ensures(__ESBMC_return_value != 0 || *out >= 2);

    *out = 5;
    return 0;
}

int parent(int *eaten)
{
    __ESBMC_requires(eaten != ((void *)0));
    __ESBMC_assigns(*eaten);
    __ESBMC_ensures(__ESBMC_return_value != 0 || *eaten > 0);
    __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == -1);

    int hdr_len = 0;
    int ret = leaf(&hdr_len);
    if (ret) {
        return -1;
    }
    *eaten = hdr_len;
    return 0;
}

int main(void)
{
    int eaten = 0;
    int ret = parent(&eaten);
    if (ret == 0) {
        assert(eaten > 0);
    }
    return 0;
}
