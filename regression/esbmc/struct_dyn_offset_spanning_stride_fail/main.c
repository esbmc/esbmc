struct S { unsigned short a, b, c, d, e; };   /* 10 bytes */
struct S s[4];
int main(void)
{
  int j = nondet_int();
  __ESBMC_assume(j == 0 || j == 3);
  unsigned int *r = (unsigned int *)s + j;   /* j=3: byte 12 = s[1] + 2 -> s[1].b, s[1].c */
  *r = 0x00020001u;
  __ESBMC_assert(j != 3 || s[1].c == 0, "expect FAIL: s[1].c == 2");
  return 0;
}
