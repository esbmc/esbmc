struct S { unsigned short a, b, c; };
struct S s[4];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 2);
  unsigned char *p = (unsigned char *)&s[i] + 4;
  *(unsigned int *)p = 0x00020001u;
  __ESBMC_assert(s[i].c == 1, "c");
  __ESBMC_assert(s[i + 1].a == 2, "next a");
  __ESBMC_assert(s[i].a == 0 && s[i].b == 0 && s[i + 1].b == 0, "untouched");
  return 0;
}
