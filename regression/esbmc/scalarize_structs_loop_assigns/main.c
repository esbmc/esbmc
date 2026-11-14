/* A loop's assigns clause over a field names the field's variable under
 * --scalarize-structs (#7521), so the frame rule havocs what the loop writes
 * and keeps the other field. */
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
  __ESBMC_loop_assigns(s.i);
  __ESBMC_loop_invariant(s.i >= 0 && s.i <= 10);
  while (s.i < s.n)
    s.i++;
  __ESBMC_assert(s.i == 10 && s.n == 10, "post");
  return 0;
}
