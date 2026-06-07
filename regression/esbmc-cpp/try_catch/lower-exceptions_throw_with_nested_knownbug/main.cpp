// std::throw_with_nested(Outer) throws a type deriving from both Outer and
// std::nested_exception, capturing the in-flight Inner via current_exception.
// The outer handler catches it as Outer&, then std::rethrow_if_nested recovers
// and re-raises the captured Inner. Outer is polymorphic (derives from
// std::exception) so rethrow_if_nested's dynamic_cast to nested_exception is
// well-formed.
//
// KNOWNBUG: MI base-subobject layout, catch-by-base, and dynamic_cast across a
// non-zero base offset are now all correct, so rethrow_if_nested's cross-cast
// works. The remaining blocker is MI base *construction* for the synthesized
// combined type: when a base (Outer) itself has a polymorphic base, the second
// base subobject (nested_exception) is not initialised at its offset, giving an
// "illegal offset" dereference. Once that base-ctor edge is fixed this verifies
// SUCCESSFUL and can move to CORE.
#include <exception>
#include <cassert>

struct Inner
{
  int v;
  Inner(int x) : v(x)
  {
  }
};
struct Outer : std::exception
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
