// Negative counterpart of github_4085_deducing_this: the explicit object
// parameter must carry the real object through, so a claim contradicting it
// is refuted rather than vacuously held (issue #4085).
struct S
{
  int v;
  int by_ref(this S &self)
  {
    return self.v + 2;
  }
};

int main()
{
  S s{10};
  __ESBMC_assert(s.by_ref() == 99, "must not hold");
  return 0;
}
