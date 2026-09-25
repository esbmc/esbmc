// `&C::m` written inline reaches symex as the address of a symbol named after
// the member, while one stored in a pointer-to-member variable arrives as a
// member_ref because renaming substitutes the variable's value. Only the
// second was resolved, so every inline `.*` / `->*` aborted with
// "pointer-to-member: constant propagation failed" (issue #6717).
struct Foo
{
  int a;
  int x;
};

struct Base
{
  int b;
};
struct Der : Base
{
  int d;
};

Foo &get(Foo &v)
{
  return v;
}

int main()
{
  Foo v;
  v.a = 1;
  v.x = 1;
  v.*(&Foo::x) = 2;
  __ESBMC_assert(v.x == 2 && v.a == 1, "writes the named member, not another");

  Foo w;
  w.x = 1;
  Foo *p = &w;
  p->*(&Foo::x) = 2;
  __ESBMC_assert(w.x == 2, "->* through a pointer");

  Foo r;
  r.x = 1;
  get(r).*(&Foo::x) = 2;
  __ESBMC_assert(r.x == 2, ".* on a reference-returning call");

  Foo s;
  s.x = 7;
  int read = s.*(&Foo::x);
  __ESBMC_assert(read == 7, "reads as well as writes");

  // Inherited members resolve through the nested base subobject.
  Der o;
  o.b = 1;
  o.*(&Der::b) = 5;
  __ESBMC_assert(o.b == 5, "inherited member via the base subobject");

  // The spelling that already worked, kept pinned alongside.
  Foo n;
  n.x = 1;
  int Foo::*pm = &Foo::x;
  n.*pm = 2;
  __ESBMC_assert(n.x == 2, "named pointer-to-member still resolves");
  return 0;
}
