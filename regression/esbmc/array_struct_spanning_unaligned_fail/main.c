struct S { unsigned short a, b, c; };
struct S s[4];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 2);
  unsigned char *p = (unsigned char *)&s[i] + 5;
  *(unsigned int *)p = 0x03020100u;   /* bytes 5..8: s[i].c hi=00, s[i+1].a=0x0201, s[i+1].b lo=03 */
  __ESBMC_assert(s[i + 1].a == 0, "expect FAIL");
  return 0;
}
