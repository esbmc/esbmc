// The sibling of github_6717_lvalue_ternary: the conditional's operands are
// reference *variables* rather than reference-returning calls. Those read as
// dereferences, so the chain reaching the dereference pass is
// `(c ? *ra : *rb).x`, and the nonscalar walk -- which accumulates an offset
// over the whole expression -- aborted on the `if` sitting in the middle of
// it. Left open by #6719; issue #6717.
extern "C" bool nondet_bool();

struct Foo
{
  int x;
};

int main()
{
  Foo a, b;
  Foo &ra = a, &rb = b;

  a.x = 1;
  b.x = 1;
  bool t = true;
  (t ? ra : rb).x = 2;
  __ESBMC_assert(a.x == 2 && b.x == 1, "the true branch writes a");

  a.x = 1;
  b.x = 1;
  bool f = false;
  (f ? ra : rb).x = 2;
  __ESBMC_assert(b.x == 2 && a.x == 1, "the false branch writes b");

  // A symbolic condition: each arm must be resolved under the guard that
  // selects it, so exactly the selected object is written.
  a.x = 1;
  b.x = 1;
  bool c = nondet_bool();
  (c ? ra : rb).x = 2;
  __ESBMC_assert(
    c ? (a.x == 2 && b.x == 1) : (b.x == 2 && a.x == 1),
    "the selected object, and only it, is written");
  return 0;
}
