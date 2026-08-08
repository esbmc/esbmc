// Negative control for github_6717_lvalue_ternary_refvars: the condition is
// false, so `b` is written and this claim about `a` must be refuted rather
// than passing vacuously (#6717).
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
  bool f = false;
  (f ? ra : rb).x = 2;
  __ESBMC_assert(a.x == 2, "must be refuted: the false branch writes b");
  return 0;
}
