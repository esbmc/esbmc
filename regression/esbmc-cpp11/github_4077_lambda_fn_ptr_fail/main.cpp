// Negative counterpart of github_4077_lambda_fn_ptr: the call through the
// pointer must carry the lambda's real effect, so a claim contradicting it is
// refuted rather than vacuously held -- which is what a bodyless invoker
// would have produced (issue #4077).
int g = 0;

int main()
{
  void (*p)() = [] { g = 7; };
  p();
  __ESBMC_assert(g == 0, "must not hold");
  return 0;
}
