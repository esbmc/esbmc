// Negative counterpart of github_6717_inline_ptr_to_member: the resolved
// member access carries the real write, so a claim contradicting it is
// refuted rather than vacuously held (issue #6717).
struct Foo
{
  int x;
};

int main()
{
  Foo v;
  v.x = 1;
  v.*(&Foo::x) = 2;
  __ESBMC_assert(v.x == 1, "must not hold");
  return 0;
}
