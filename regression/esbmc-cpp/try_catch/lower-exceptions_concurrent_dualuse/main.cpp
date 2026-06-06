#include <pthread.h>
#include <cassert>

// worker is used BOTH as a pthread start routine and called directly. The
// lowering enforces a thread's uncaught-escape terminate at the start routine's
// own epilogue, which would be wrong for the direct call (an escape there should
// propagate to the direct caller, not terminate). The pass cannot tell the two
// call contexts apart per-function, so it conservatively falls back to the
// imperative path for the whole program. Each use catches its own exception, so
// the (imperative) verdict is SUCCESSFUL.
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
