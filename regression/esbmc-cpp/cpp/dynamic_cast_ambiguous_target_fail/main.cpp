// [expr.dynamic.cast]/8: both casts fail when T occurs twice in D.
#include <cassert>
#include <typeinfo>
struct T { int t = 5; virtual ~T() {} };
struct X : T { };
struct Y : T { };
struct S { int s = 7; virtual ~S() {} };
struct D : S, X, Y { };
struct E : S, X { };
bool nondet_bool();
int main()
{
  D d;
  E e;
  bool ambiguous = nondet_bool();
  S *s = ambiguous ? static_cast<S *>(&d) : static_cast<S *>(&e);
  T *t = dynamic_cast<T *>(s);
  assert(t != nullptr);
  if (!ambiguous)
    assert(t->t == 5);

  bool thrown = false;
  try
  {
    (void)dynamic_cast<T &>(*s);
  }
  catch (std::bad_cast &)
  {
    thrown = true;
  }
  assert(!thrown);
  return 0;
}
