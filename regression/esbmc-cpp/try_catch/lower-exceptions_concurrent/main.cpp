#include <pthread.h>

// Exception state is a single global tuple, so the lowered dispatch is unsound
// across threads. Lowering is the only exception path (the legacy imperative
// path was removed, #5075), so a concurrent program that uses exceptions is
// reported as a hard error rather than miscompiled.
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
