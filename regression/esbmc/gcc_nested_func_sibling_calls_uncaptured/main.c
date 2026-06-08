// Nested `a` calls sibling `c` which captures nothing.  The call to
// `c()` must NOT receive any injected arguments — spurious args would
// cause a signature mismatch and compile failure under Clang's real
// parse pass.
int main()
{
  int seen = 0;
  int c(void) { return 1; }
  void a(void) { seen = c(); }
  a();
  __ESBMC_assert(seen == 1, "uncaptured sibling: no stray args");
  return 0;
}
