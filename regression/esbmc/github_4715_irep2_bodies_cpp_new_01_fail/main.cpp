// Negative companion to github_4715_irep2_bodies_cpp_new_01.
//
// Asserts the *wrong* value for the scalar-new initializer: `new int(7)` must
// yield 7, so `*p == 0` is violated and verification FAILS. This pins the
// initializer fix — if the round-trip ever drops the initializer again the
// object default-inits to 0, this assertion would spuriously hold and the
// expected FAILED verdict would flip, catching the regression.
int main()
{
  int *p = new int(7);
  __ESBMC_assert(*p == 0, "initializer must NOT be dropped to 0");
  delete p;
  return 0;
}
