struct S { unsigned short a, b, c; };
struct S s[4];
int main(void)
{
  int j = nondet_int();
  __ESBMC_assume(j == 0 || j == 1);
  s[1].b = 1; s[1].c = 2; s[2].a = 3; s[2].b = 4;
  unsigned long long *r = (unsigned long long *)s + j;
  unsigned long long v = *r;
  __ESBMC_assert(j != 1 || v == 0x0004000300020001ull, "read");
  return 0;
}
