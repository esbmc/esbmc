// __ESBMC_assert is a built-in intrinsic; no include needed.

// W1-loc Phase C (esbmc/esbmc#4715): under --irep2-native-body the
// code_expression2t handler now delegates a code cpp-throw operand to the
// legacy convert() (as convert_expression's is_code branch does), so f() -- an
// `if` guarding a `throw` -- converts natively instead of falling back on the
// throw. main()'s try/catch is not yet a native kind, so main falls back; the
// thrown value must still propagate and be caught.
int f(int x)
{
  if (x < 0)
    throw x;
  return x;
}

int main()
{
  int r = 0;
  try
  {
    r = f(-5);
  }
  catch (int e)
  {
    r = e;
  }
  __ESBMC_assert(r == -5, "caught the thrown value");
  return 0;
}
