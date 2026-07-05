#include <pthread.h>
#include <cassert>

// worker is used BOTH as a pthread start routine and called directly. The
// lowering enforces a thread's uncaught-escape terminate at the start routine's
// own epilogue, which would be wrong for the direct call (an escape there should
// propagate to the direct caller, not terminate). The pass cannot tell the two
// call contexts apart per-function, so it conservatively declines the whole
// program as unsupported (the imperative fallback was removed in #5244).
// KNOWNBUG: sound call-site-sensitive enforcement at the pthread trampoline is
// not yet implemented; each use catches its own exception, so the intended
// verdict is SUCCESSFUL once that lands.
struct E
{
  int v;
  E(int x) : v(x)
  {
  }
};

void *worker(void *arg)
{
  int id = *(int *)arg;
  try
  {
    throw E(id);
  }
  catch (E &e)
  {
    assert(e.v == id);
  }
  return 0;
}

int main()
{
  int a = 0;
  worker(&a); // direct call
  pthread_t t;
  pthread_create(&t, 0, worker, &a); // also a thread start routine
  return 0;
}
