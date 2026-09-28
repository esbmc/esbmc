// Specialisations over same-named local classes of two functions stay distinct
// through every argument shape: pointer, array, reference, pack, function
// type, member of a specialisation, and a declaration argument.
#include <cassert>
template <class T> struct Box { T t; };
template <class T> struct Ref { T r; };
template <class... Ts> struct Pack;
template <class T> struct Pack<int, T> { T v; };
template <class F> struct Fn;
template <class R> struct Fn<R (*)()> { R r; };
template <class T> struct Outer { struct In { T t; }; };
template <auto *P> struct Addr { decltype(P) p = P; };

int f()
{
  struct S { int a; };
  static S s;
  S x{};
  Box<S *> bp{&x};
  Box<S[1]> ba{};
  Ref<S &> rr{x};
  Pack<int, S> pk{};
  Fn<S (*)()> fn{};
  Outer<S>::In in{};
  Addr<&s> ad;
  return bp.t->a + ba.t[0].a + rr.r.a + pk.v.a + fn.r.a + in.t.a + ad.p->a;
}

int g()
{
  struct S { int a; int b; int c; };
  static S s;
  s.c = 1;
  S x{};
  x.c = 1;
  Box<S *> bp{&x};
  Box<S[1]> ba{};
  ba.t[0].c = 1;
  Ref<S &> rr{x};
  Pack<int, S> pk{};
  pk.v.c = 1;
  Fn<S (*)()> fn{};
  fn.r.c = 1;
  Outer<S>::In in{};
  in.t.c = 1;
  Addr<&s> ad;
  return bp.t->c + ba.t[0].c + rr.r.c + pk.v.c + fn.r.c + in.t.c + ad.p->c;
}

int main()
{
  assert(f() == 0);
  assert(g() != 7);
  return 0;
}
