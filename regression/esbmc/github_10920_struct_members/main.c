/* esbmc/esbmc#10920: as github_10920_array_stores, for struct members. */
struct pair
{
  int a;
  int b;
};
int nondet_int();
int main()
{
  struct pair p;
  p.a = nondet_int();
  p.b = nondet_int();
  __ESBMC_assume(p.a >= 0 && p.a <= 3 && p.b >= 0 && p.b <= 3);
  __ESBMC_assert(p.a + p.b != 3, "p.a + p.b != 3");
  return 0;
}
