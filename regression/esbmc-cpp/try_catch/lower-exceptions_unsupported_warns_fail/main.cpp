// Same unsupported empty-catch-all layout as lower-exceptions_unsupported_warns,
// but an assertion fails after the handler, so the (imperative-fallback) verdict
// is FAILED. The lowering still reports the fallback diagnostic rather than
// lowering silently (#5075, P4 prerequisite).
#include <cassert>

int main()
{
  try
  {
    throw 1;
  }
  catch (...)
  {
  }
  assert(0);
}
