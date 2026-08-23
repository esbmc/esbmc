/* github_6212_reference_extent_pass:
 * A reference parameter is lowered to a pointer, but C++ guarantees it binds
 * to one complete object, so it keeps a one-element backing like the implicit
 * receiver. There is no migration to state an extent for it either:
 * __ESBMC_is_fresh needs a pointer to point at fresh storage, and the contract
 * of f has none to name.
 */
struct S
{
  int n;
};

void f(S &s)
{
  __ESBMC_requires(s.n >= 0);
  __ESBMC_assigns(s.n);
  __ESBMC_ensures(s.n == 1);
  s.n = 1;
}

int main()
{
  return 0;
}
