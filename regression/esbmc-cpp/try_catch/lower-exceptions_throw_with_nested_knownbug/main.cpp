// std::throw_with_nested(Outer) throws a type deriving from both Outer and
// std::nested_exception, capturing the in-flight Inner via current_exception.
// The outer handler catches it as Outer&, then std::rethrow_if_nested recovers
// and re-raises the captured Inner.
//
// KNOWNBUG: the thrown combined type is multiple-inheritance (Outer +
// polymorphic nested_exception). When a non-polymorphic base precedes the
// polymorphic one, the catch-by-base binding reads the base subobject at the
// wrong offset, so `o.w` reads garbage and the assertion spuriously fails. This
// is a pre-existing base-subobject offset bug (construction vs catch) in the
// C++ frontend/lowering, independent of throw_with_nested itself. Once the MI
// base-offset handling is fixed this verifies SUCCESSFUL and can move to CORE.
#include <exception>
#include <cassert>

struct Inner
{
  int v;
  Inner(int x) : v(x)
  {
  }
};
struct Outer
{
  int w;
  Outer(int x) : w(x)
  {
  }
};

void f()
{
  try
  {
    throw Inner(1);
  }
  catch (...)
  {
    std::throw_with_nested(Outer(2));
  }
}

int main()
{
  try
  {
    f();
  }
  catch (Outer &o)
  {
    assert(o.w == 2);
    try
    {
      std::rethrow_if_nested(o);
    }
    catch (Inner &i)
    {
      assert(i.v == 1);
      return 0;
    }
    assert(0); // rethrow_if_nested should have re-raised Inner
  }
  assert(0);
  return 0;
}
