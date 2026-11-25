/* #11014: a store spanning two int members under --ir reaches both */
struct s { int a; int b; };
int main(void)
{
  struct s s = {1, 2};
  long long *p = (long long *)&s;
  *p = 0x0000000700000005LL;
  __ESBMC_assert(s.a == 5 && s.b == 7, "store reaches both members");
  __ESBMC_assert(*p == 0x0000000700000005LL, "read back");
  return 0;
}
