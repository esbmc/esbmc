// An lvalue conditional denotes an object, not a copy of one. The frontend
// typed the `if` from clang's getType(), which drops the reference, so both
// branches were dereferenced into a temporary and the assignment landed on
// that copy -- leaving the original untouched and the claim below reported as
// violated. Reduced from g++.dg/expr/cond18.C (issue #6717).
struct Foo
{
  int x;
};

Foo &get(Foo &v)
{
  return v;
}

int main()
{
  bool c = true;

  Foo v;
  v.x = 1;
  (c ? get(v) : get(v)).x = 2;
  __ESBMC_assert(v.x == 2, "the write reaches the referenced object");

  // The condition still selects: only the taken branch's object is written.
  Foo a, b;
  a.x = 1;
  b.x = 1;
  bool f = false;
  (f ? get(a) : get(b)).x = 2;
  __ESBMC_assert(b.x == 2 && a.x == 1, "the false branch writes b, not a");

  // Object- and scalar-typed conditionals were never affected; keep them
  // pinned so the reference case cannot be fixed at their expense.
  Foo p, q;
  p.x = 1;
  q.x = 1;
  (c ? p : q).x = 2;
  __ESBMC_assert(p.x == 2, "object conditional still assigns through");

  int i = 1, j = 1;
  (c ? i : j) = 2;
  __ESBMC_assert(i == 2, "scalar conditional still assigns through");
  return 0;
}
