/* github #9225: subtracting pointers to elements of size 1 or 2^k needs no
 * divider. */
unsigned nondet_uint(void);
int a[10];
char c[10];

int main(void)
{
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 10);
  __ESBMC_assert(&a[i] - a + (&c[i] - c) == 2 * i, "");
  return 0;
}
