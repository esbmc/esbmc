/* is_fresh_requires_char_pass: validate the char-pointer (alignment-1) path.
 * Even with a byte-aligned pointee, the broken else branch used to fire
 * 'invalid pointer' on the canonical form; this test exercises that case.
 */
__ESBMC_contract
char foo(char *p)
{
    __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(char)));
    __ESBMC_ensures(1);

    return *p;
}

int main(void) { return 0; }
