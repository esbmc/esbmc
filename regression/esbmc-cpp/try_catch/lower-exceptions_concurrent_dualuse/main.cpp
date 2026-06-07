#include <pthread.h>
#include <cassert>

// worker is used BOTH as a pthread start routine and called directly. The
// lowering enforces a thread's uncaught-escape terminate at the start routine's
// own epilogue; marking worker an entry is a sound over-approximation (at worst
// a spurious terminate on an escape the direct caller would catch, never a
// missed bug). Here each use catches its own exception, so worker never escapes
// and the verdict is SUCCESSFUL.
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
