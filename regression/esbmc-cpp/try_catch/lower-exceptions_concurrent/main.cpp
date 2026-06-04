#include <pthread.h>

// Exception state is a single global tuple, so the lowered dispatch is unsound
// across threads; the pass must leave concurrent programs to the imperative
// path. Each thread catches its own exception locally.
struct E
{
};

void *worker(void *)
{
  try
  {
    throw E();
  }
  catch (E &)
  {
  }
  return 0;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, worker, 0);
  pthread_create(&t2, 0, worker, 0);
  return 0;
}
