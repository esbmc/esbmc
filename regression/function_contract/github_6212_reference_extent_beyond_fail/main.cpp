/* github_6212_reference_extent_beyond_fail:
 * Negative control for github_6212_reference_extent_pass. One element is all
 * the reference promises, so an access past it must still be caught.
 */
struct S
{
  int n;
};

void f(S &s)
{
  __ESBMC_requires(s.n >= 0);
  __ESBMC_ensures(1);
  (&s)[3].n = 1;
}

int main()
{
  return 0;
}
