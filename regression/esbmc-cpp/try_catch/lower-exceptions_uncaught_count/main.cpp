// std::uncaught_exceptions() reports the per-thread count of exceptions thrown
// (or rethrown) but not yet entered into their handler ([except.uncaught]).
// The GOTO exception lowering maintains the count: +1 at a throw/rethrow, -1
// when a handler is entered. With no active exception the count is 0; it is 0
// again once an exception is caught, and a rethrow makes it uncaught anew.
#include <exception>
#include <cassert>

int main()
{
  assert(std::uncaught_exceptions() == 0);
  try
  {
    assert(std::uncaught_exceptions() == 0);
    try
    {
      throw 1; // one exception now uncaught
    }
    catch (int)
    {
      assert(std::uncaught_exceptions() == 0); // entered handler -> 0
      throw;                                   // rethrow -> uncaught again
    }
  }
  catch (int)
  {
    assert(std::uncaught_exceptions() == 0); // re-caught -> 0
  }
  assert(std::uncaught_exceptions() == 0);
  return 0;
}
