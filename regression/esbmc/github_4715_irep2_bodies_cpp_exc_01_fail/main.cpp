// Negative variant of github_4715_irep2_bodies_cpp_exc_01: the exception is
// caught by the matching handler, but the in-handler assertion is wrong, so
// ESBMC must falsify it. This proves the catch-handler body and its
// exception_id are faithfully reconstructed across the --irep2-bodies body
// round-trip (a dropped handler would yield a spurious SUCCESSFUL).
int main()
{
  try
  {
    throw 42;
  }
  catch (int e)
  {
    __ESBMC_assert(e == 7, "caught value is (wrongly) 7");
  }
  return 0;
}
