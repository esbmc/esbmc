// Inner-block declaration shadows outer x; outer x is not used inside
// the nested function.  A buggy transform would treat outer x as captured
// and rewrite `int x = 99;` to a garbage pointer-type decl that does not
// compile.  Mutating outer x after the decl and before the call makes
// that rewrite observable: if it happened the file would fail Clang parse
// at the GOTO conversion step; with the correct transform the assertion
// holds.
int main()
{
  int x = 7;
  int inner(void)
  {
    {
      int x = 99;
      return x;
    }
  }
  x = 50;
  __ESBMC_assert(inner() == 99, "inner block decl not rewritten as capture");
  (void)x;
  return 0;
}
