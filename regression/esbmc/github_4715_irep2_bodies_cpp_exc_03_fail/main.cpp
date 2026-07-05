// Negative variant of github_4715_irep2_bodies_cpp_exc_03: the thrown class
// temporary is constructed with x == 42 and caught by reference, but the
// in-handler assertion wrongly expects 7, so ESBMC must falsify it. This proves
// the constructed value genuinely flows into the handler across the
// --irep2-bodies body round-trip — a vacuous (unconstructed / havoced) object
// would let `a.x == 7` hold and yield a spurious SUCCESSFUL.
struct A
{
  int x;
  A(int v) : x(v)
  {
  }
};

int main()
{
  try
  {
    throw A(42);
  }
  catch (A &a)
  {
    __ESBMC_assert(a.x == 7, "caught value is (wrongly) 7");
    return 1;
  }
  return 0;
}
