// Negative counterpart of github_4078_cond_decl: the condition variable's
// conversion to bool must carry its real value, so a claim contradicting it
// is refuted rather than vacuously held (issue #4078).
struct c
{
  int a;
  c(int x) : a(x)
  {
  }
  operator bool()
  {
    return a != 0;
  }
};

int main()
{
  int hit = 0;
  if (c b = 1)
    hit = 1;
  __ESBMC_assert(hit == 0, "must not hold");
  return 0;
}
