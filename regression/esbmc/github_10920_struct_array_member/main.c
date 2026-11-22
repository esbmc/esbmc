/* esbmc/esbmc#10920: an array inside a struct input is blocked element by
   element too. */
struct s
{
  int v[2];
};
int main()
{
  struct s p;
  __ESBMC_assume(p.v[0] >= 0 && p.v[0] <= 3 && p.v[1] >= 0 && p.v[1] <= 3);
  __ESBMC_assert(p.v[0] + p.v[1] != 3, "p.v[0] + p.v[1] != 3");
  return 0;
}
