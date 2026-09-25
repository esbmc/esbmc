// std::throw_with_nested(Outer) throws a type deriving from both Outer and
// std::nested_exception, capturing the in-flight Inner via current_exception.
// The outer handler catches it as Outer&, then calls std::rethrow_if_nested(o).
//
// `Outer` here is NON-polymorphic. Per [except.nested]/6, rethrow_if_nested is a
// no-op unless the *static* type of its argument is a polymorphic class (the
// underlying dynamic_cast<nested_exception*> is otherwise ill-formed, so the
// library selects the empty overload — see __is_polymorphic in <exception>).
// So the captured Inner is NOT re-raised, the inner catch(Inner&) never fires,
// and the trailing assert(0) is reachable. This matches real C++: g++/clang++
// compile and run this program to that assert(0) (abort). ESBMC is thus
// correct to report VERIFICATION FAILED; the previous SUCCESSFUL expectation and
// its "base-offset bug" note were wrong (`o.w == 2` passes -- Outer is the first
// base, at offset 0). A positive recovery test needs a *polymorphic* Outer, which
// exercises the MI dynamic_cast cross-cast handled separately.
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
