struct I { unsigned short u, v; };
struct S { unsigned short a; struct I in; };
struct S s[4];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 2);
  unsigned char *p = (unsigned char *)&s[i] + 4;
  *(unsigned int *)p = 0x00090008u;
  __ESBMC_assert(s[i].in.v == 8, "v");
  __ESBMC_assert(s[i + 1].a == 9, "next a");
  return 0;
}
