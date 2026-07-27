// __ESBMC_assert is a built-in intrinsic; no include needed.

// W1-loc Phase C (esbmc/esbmc#4715): under --irep2-native-body a source-level
// try/catch (code_cpp_catch2t) is delegated to the legacy convert()/convert_catch
// rather than forcing a whole-function fallback, so the statements around it --
// the local declarations and the trailing return here -- convert natively while
// convert_catch still owns the CATCH markers, handler targets and stack unwind.
int f(int x)
{
  if (x < 0)
    throw x;
  return x;
}

int compute(int a)
{
  int base = 100;
  int r;
  try
  {
    r = f(a);
  }
  catch (int e)
  {
    r = -e;
  }
  return base + r;
}

int main()
{
  __ESBMC_assert(compute(-5) == 0, "must fail: exception was caught and combined");
  return 0;
}
