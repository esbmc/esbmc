#include <pthread.h>
#include <cassert>

// Negative counterpart to lower-exceptions_concurrent: each worker throws E(id)
// and the handler reads its own thread's object, but asserts e.v == 0. The
// thread launched with id 1 catches E(1), so e.v == 0 is false and the property
// is violated -> VERIFICATION FAILED. Confirms the lowered per-thread dispatch
// still detects genuine failures (it does not trivially verify everything).
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
    assert(e.v == 0); // fails for the thread launched with id 1
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
