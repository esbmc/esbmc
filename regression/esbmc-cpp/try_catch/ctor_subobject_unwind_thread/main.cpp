// A program with no throw gets no handler for a constructor's subobjects, so
// it stays out of exception lowering, which declines a thread start routine
// that is also called directly.
#include <cassert>
#include <pthread.h>

struct M
{
  int v;
  M() : v(1)
  {
  }
  ~M()
  {
  }
};

struct W
{
  M a;
  M b;
  W() : a(), b()
  {
  }
};

void *worker(void *)
{
  W w;
  assert(w.b.v == 1);
  return nullptr;
}

int main()
{
  pthread_t t;
  pthread_create(&t, nullptr, worker, nullptr);
  worker(nullptr);
  pthread_join(t, nullptr);
  return 0;
}
