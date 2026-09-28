// __ESBMC_assert is a built-in intrinsic; no include needed.

// Negative variant of github_4715_irep2_native_body_temp_dtor_01: the
// temporary's destructor DOES run across the native return fallback, so
// asserting it did not run must fail.
int dtor_calls = 0;

struct M
{
  int w;
  M(int a) : w(a) {}
  ~M() { dtor_calls++; }
};

int use(const M &m)
{
  return m.w;
}

int g;

int check()
{
  g = use(M(1));
  return 5;
}

int main()
{
  check();
  __ESBMC_assert(dtor_calls == 0, "must fail: temporary destructor ran");
  return 0;
}
