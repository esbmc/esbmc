// The shapes either side of the rejection in
// github_6717_throw_conditional_reject: a scalar conditional never
// materialises, so a throw branch is fine there, and a class-typed
// conditional without a throw was never affected (issue #6717).
struct S
{
  int v;
  S(int p) : v(p)
  {
  }
};

int scalar(int x)
{
  return x ? throw 1 : 5;
}

S pick(bool c)
{
  S a(1), b(2);
  return c ? a : b;
}

int main()
{
  __ESBMC_assert(scalar(0) == 5, "throw in a scalar conditional");
  __ESBMC_assert(pick(true).v == 1, "class conditional without a throw");
  __ESBMC_assert(pick(false).v == 2, "and its other branch");
  return 0;
}
