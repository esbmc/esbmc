struct S { unsigned short a; unsigned short b : 4, c : 12; unsigned short d; };
struct S s[4];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 2);
  unsigned char *p = (unsigned char *)&s[i] + 4;
  *(unsigned int *)p = 0x00020001u;
  __ESBMC_assert(s[i].d == 1, "d");
  __ESBMC_assert(s[i + 1].a == 2, "next a");
  return 0;
}
