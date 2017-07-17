int main(void)
{
    unsigned int b;
    unsigned int a = 0;
    int table[2][2]  = {{1,0}, {1,0}};

    if (nondet_uint() == 0) {
        a = 1;
    } else {
        __ESBMC_assume(0);
    }
    __ESBMC_assert(table[1][0], "FOO"); /* Assertion ok */
    __ESBMC_assert(a == 1, "FOO"); /* Assertion ok */
    b = table[a][0];
    __ESBMC_assert(b, "ERROR"); /* Assertion violated */
    __ESBMC_assert(table[a][0], "ERROR"); /* arrays bound violated */
    return (0);
}
