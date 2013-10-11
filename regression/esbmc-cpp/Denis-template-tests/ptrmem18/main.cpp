#include <cassert>
// PR c++/33616
// { dg-do run }
// { dg-options "-O2" }

struct S {
  int c;
  S () : c (0) {}
  virtual void f1 () { c += 1; }
  virtual void f2 () { c += 16; }
};

struct T {
  S s;
};

typedef void (S::*Q) ();

template <Q P>
void test1 (T *t)
{
  (t->s.*P)();
}

template <Q P>
void test2 (T *t)
{
  S &s = t->s;
  (s.*P)();
}

int
main ()
{
  T t;
  test1 <&S::f1> (&t);
  if (t.s.c != 1)
    assert(0);
  test1 <&S::f2> (&t);
  if (t.s.c != 17)
    assert(0);
  test2 <&S::f1> (&t);
  if (t.s.c != 18)
    assert(0);
  test2 <&S::f2> (&t);
  if (t.s.c != 34)
    assert(0);
}
