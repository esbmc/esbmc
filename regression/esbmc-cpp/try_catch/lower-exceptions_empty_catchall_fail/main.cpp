// Same empty-catch-all layout as lower-exceptions_empty_catchall (lowered in-
// line via normalize_empty_handlers, no fallback), but an assertion fails after
// the handler, so the lowered-path verdict is FAILED (#5075).
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
