// Same elided-skip-GOTO empty catch-all as lower-exceptions_empty_catchall, but
// an assertion fails on the post-handler path. The catch-all lowers and clears
// the in-flight exception, control resumes after the (empty) handler, and the
// assertion is reached — so verification is FAILED (#5075).
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
