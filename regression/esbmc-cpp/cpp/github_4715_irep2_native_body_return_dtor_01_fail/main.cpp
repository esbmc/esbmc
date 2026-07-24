// __ESBMC_assert is a built-in intrinsic; no include needed.

// W1-loc Phase C (esbmc/esbmc#4715): under --irep2-native-body a `return` that
// runs while a local object's destructor is still on the stack is delegated to
// the legacy convert_return, which captures the value into a temporary, runs the
// destructor (C++ [stmt.return]: after the value is computed, before the jump)
// and returns the temporary. Before this the whole function fell back on such a
// return; now probe() -- a local object followed by a value return -- converts
// natively, and the destructor still runs exactly once.
int dtor_count = 0;

struct T
{
  int v;
  T(int a) : v(a) {}
  ~T() { dtor_count++; }
};

int probe(int x)
{
  T t(x);
  return t.v * 2;
}

int main()
{
  int r = probe(5);
  __ESBMC_assert(r == 10 && dtor_count == 0, "must fail: destructor ran once");
  return 0;
}
