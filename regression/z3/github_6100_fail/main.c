_Bool eq(int a, int b)
{
  return a == b;
}

int main()
{
  int var;
  // No value satisfies var == var + 1, so the existential is false and the
  // assertion must fail. Before the fix, eq() was hoisted out of the
  // quantifier and evaluated once, masking such cases (discussion #6100).
  __ESBMC_assert(__ESBMC_exists(&var, eq(var, var + 1)), "exists-false");
}
