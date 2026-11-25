/* A loop invariant over fields reads the field variables under
 * --scalarize-structs (#7521), not the struct they were split from. */
struct S
{
  int i;
  int n;
};

int main(void)
{
  struct S s;
  s.i = 0;
  s.n = 10;
  __ESBMC_loop_invariant(s.i >= 0 && s.i <= 10 && s.n == 10);
  while (s.i < s.n)
    s.i++;
  __ESBMC_assert(s.i == 10, "post");
  return 0;
}
