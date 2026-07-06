// A catch parameter that names a function type (`exception()`) is ill-formed
// but accepted by the parser; it produced an empty exception-id vector and
// crashed the adjuster (ids.front() on an empty vector). It can never match a
// real throw, so the thrown int is handled by the following catch.
#include <exception>
#include <cassert>
using std::exception;

int main()
{
  int caught = 0;
  try
  {
    throw 42;
  }
  catch (exception())
  {
    caught = -1;
  }
  catch (int e)
  {
    caught = e;
  }
  assert(caught == 42);
  return 0;
}
