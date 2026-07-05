// Negative companion to lower-exceptions_uncaught_count: once an exception has
// entered its handler it is no longer uncaught, so std::uncaught_exceptions()
// is 0 inside the catch, not 1. Asserting the wrong count must fail.
#include <exception>
#include <cassert>

int main()
{
  try
  {
    throw 1;
  }
  catch (int)
  {
    assert(std::uncaught_exceptions() == 1); // wrong: the count is 0 here
  }
  return 0;
}
