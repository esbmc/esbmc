#include <pthread.h>
#include <cassert>

// The exception-state globals (and the thrown-object storage) are thread-local,
// so the lowered dispatch is sound across threads: each thread raises, catches,
// and reads its OWN in-flight exception. Each worker throws E(id) and the
// handler must observe its own id — a shared object would let one thread read
// another's, which the per-thread storage rules out.
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
  pthread_t t1, t2;
  int a = 0, b = 1;
  pthread_create(&t1, 0, worker, &a);
  pthread_create(&t2, 0, worker, &b);
  return 0;
}
