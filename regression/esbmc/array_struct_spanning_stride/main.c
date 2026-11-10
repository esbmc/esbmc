struct S { unsigned short a, b, c; };
struct S s[4];
int main(void)
{
  unsigned long long *q = (unsigned long long *)s;
  int j = nondet_int();
  __ESBMC_assume(j == 0 || j == 1);
  q[j] = 0x0004000300020001ull;
  __ESBMC_assert(j != 1 || (s[1].b == 1 && s[1].c == 2 && s[2].a == 3 && s[2].b == 4), "exact");
  return 0;
}
