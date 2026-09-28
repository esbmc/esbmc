// Negative counterpart of github_6717_lvalue_ternary: the write really does
// reach the referenced object, so a claim that it did not -- which is exactly
// what the old copy-to-temporary lowering made true -- is refuted (issue
// #6717).
struct Foo
{
  int x;
};

Foo &get(Foo &v)
{
  return v;
}

int main()
{
  bool c = true;
  Foo v;
  v.x = 1;
  (c ? get(v) : get(v)).x = 2;
  __ESBMC_assert(v.x == 1, "must not hold");
  return 0;
}
