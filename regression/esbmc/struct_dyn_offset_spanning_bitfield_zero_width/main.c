struct S { unsigned int a : 4; unsigned int : 0; unsigned int c; };
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  *(unsigned long long *)&s[i] = 0x0000000700000005ull;
  __ESBMC_assert(s[i].a == 5 && s[i].c == 7, "gap");
  return 0;
}
