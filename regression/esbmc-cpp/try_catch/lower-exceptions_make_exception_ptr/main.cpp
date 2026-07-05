// std::make_exception_ptr(e) captures e into an exception_ptr by throwing and
// catching it (current_exception), and std::rethrow_exception re-raises it. The
// value carried through the slot must be preserved. Exercises the fix for
// throwing a by-value operand: the exception object's copy/move constructor is
// now called with the source bound to its reference parameter (an address),
// rather than a struct value.
#include <exception>
#include <cassert>

struct E
{
  int v;
  E(int x) : v(x)
  {
  }
};

int main()
{
  std::exception_ptr p = std::make_exception_ptr(E(42));
  assert((bool)p);
  try
  {
    std::rethrow_exception(p);
  }
  catch (E &e)
  {
    assert(e.v == 42);
    return 0;
  }
  assert(0); // must have been caught
  return 0;
}
