// Negative counterpart of github_4084_consteval: the consteval result must
// reach the model as its real value, so a contradicting claim is refuted
// rather than vacuously held (issue #4084).
consteval int sq(int x)
{
  return x * x;
}

int main()
{
  constexpr int k = sq(5);
  __ESBMC_assert(k == 26, "must not hold");
  return 0;
}
