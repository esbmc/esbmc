// __ESBMC_assert is a built-in intrinsic; no include needed.

// W1-loc Phase C (esbmc/esbmc#4715): the --irep2-native-body code_expression2t
// handler now lowers a full-expression temporary_object natively (the guard
// that previously fell back was over-conservative). A result-used temporary
// whose ~M is deferred to the enclosing scope's exit leaves a destructor
// FUNCTION_CALL on the destructor stack when the native code_return2t handler
// runs; that handler reproduces only the plain, destructor-free RETURN, so it
// must fall back the moment such an entry is present. Without that fallback the
// temporary's destructor is dropped and dtor_calls stays 0.
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

// The M(1) temporary is deferred to check()'s scope exit, so its ~M call is
// still on the destructor stack at `return 5;` -> the native return handler
// falls back and legacy runs the destructor.
int check()
{
  g = use(M(1));
  return 5;
}

int main()
{
  check();
  __ESBMC_assert(dtor_calls == 1, "temporary destructor ran across return");
  return 0;
}
