// This __VERIFIER_assert also counts its calls, so the loop is not a check
// loop: one call at a nondet index would lose nine increments.
extern void abort(void);
void reach_error() { __ESBMC_assert(0, "reach_error"); }
int cnt;
void __VERIFIER_assert(int c)
{
  if (!c)
  {
    reach_error();
    abort();
  }
  cnt++;
}
int a[10];
int main()
{
  for (int i = 0; i < 10; i++)
    __VERIFIER_assert(a[i] == 0);
  __VERIFIER_assert(cnt != 10);
}
