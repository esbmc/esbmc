// Exercises C++ try/catch under --irep2-bodies (V.4.3, esbmc#4715). The
// cpp-catch node carries its try block and catch-handler blocks as operands,
// plus a per-handler exception_id. code_cpp_catch2t historically stored only
// the catchable-type list, so the body round-trip dropped the operands and
// convert_catch crashed on an empty catch. With the operands carried, the
// exception is thrown, caught by the matching handler, and the in-handler
// assertion holds, so verification must SUCCEED.
int main()
{
  try
  {
    throw 42;
    __ESBMC_assert(0, "unreachable after throw");
  }
  catch (int e)
  {
    __ESBMC_assert(e == 42, "caught value is 42");
  }
  return 0;
}
