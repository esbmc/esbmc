// Same unsupported concurrent program as lower-exceptions_unsupported_warns,
// but an assertion fails after the handler, so the (imperative-fallback) verdict
// is FAILED. The lowering still reports the fallback diagnostic rather than
// lowering silently (#5075, P4 prerequisite).
#include <pthread.h>
#include <cassert>

void *worker(void *)
{
  return 0;
}

int main()
{
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  try
  {
    throw 1;
  }
  catch (int)
  {
  }
  assert(0);
}
