/*
 * Test: Contract propagation with assigns clause — enforce parent + replace leaf
 * Expected: VERIFICATION SUCCESSFUL
 * Status: KNOWNBUG — same issue as replace_enforce_propagation_pass
 *
 * This test focuses on the assigns clause interaction:
 *   - leaf(buf, len_out): reads buf[0..1], writes *len_out = buf[0] + buf[1]
 *     Contract: assigns *len_out; ensures *len_out >= 2
 *   - parent(buf, eaten): calls leaf, assigns *eaten = len_out
 *     Contract: assigns *eaten; ensures *eaten > 0
 *
 * The assigns clause should cause havoc of *len_out in the replacement,
 * and the ensures should then constrain the havoc'd value.
 *
 * Bug: Same as replace_enforce_propagation_pass — the enforce wrapper's
 *   ensures check dereferences *eaten through a harness-generated pointer
 *   that doesn't point to valid memory when --function parent is used.
 */
#include <assert.h>

int leaf(const unsigned char *buf, int *len_out)
{
    __ESBMC_requires(buf != ((void *)0));
    __ESBMC_requires(len_out != ((void *)0));
    __ESBMC_assigns(*len_out);
    __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == -1);
    __ESBMC_ensures(__ESBMC_return_value != 0 || *len_out >= 2);

    *len_out = buf[0] + buf[1];
    if (*len_out < 2)
        return -1;
    return 0;
}

int parent(const unsigned char *buf, int *eaten)
{
    __ESBMC_requires(buf != ((void *)0));
    __ESBMC_requires(eaten != ((void *)0));
    __ESBMC_assigns(*eaten);
    __ESBMC_ensures(__ESBMC_return_value != 0 || *eaten > 0);
    __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == -1);

    int len = 0;
    int ret = leaf(buf, &len);
    if (ret) {
        return -1;
    }
    *eaten = len;
    return 0;
}

int main(void)
{
    unsigned char buf[4] = {3, 5, 0, 0};
    int eaten = 0;
    int ret = parent(buf, &eaten);
    if (ret == 0) {
        assert(eaten > 0);
    }
    return 0;
}
