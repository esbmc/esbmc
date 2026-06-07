// std::throw_with_nested(Outer) throws a type deriving from both Outer and
// std::nested_exception, capturing the in-flight Inner via current_exception.
// The outer handler catches it as Outer&, then std::rethrow_if_nested recovers
// and re-raises the captured Inner. Outer is polymorphic (derives from
// std::exception) so rethrow_if_nested's dynamic_cast to nested_exception is
// well-formed.
//
// KNOWNBUG: rethrow_if_nested needs a dynamic_cast across the multiple-
// inheritance combined type (Outer subobject -> sibling nested_exception base,
// which sits at a non-zero offset). ESBMC's MI base-subobject layout is now
// correct for casts/ctors/catch (the earlier offset-0 overlap bug is fixed),
// but dynamic_cast with a non-zero base offset is still explicitly unsupported
// ("dynamic_cast: multiple inheritance with non-zero base offset ... is not
// supported"). Once MI dynamic_cast lands this verifies SUCCESSFUL and can move
// to CORE.
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
