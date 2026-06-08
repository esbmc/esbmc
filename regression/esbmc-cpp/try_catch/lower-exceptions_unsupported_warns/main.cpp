// A concurrent program cannot lower: the exception state is one global tuple,
// not per-thread, so the lowered dispatch is unsound for threads (one thread
// could observe or clear another's in-flight exception). The pass declines and
// reports the fallback diagnostic instead of lowering silently; the imperative
// path still catches the throw, so this verifies SUCCESSFUL (#5075, P4
// prerequisite — the diagnostic is what lets the imperative path be removed).
#include <pthread.h>

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
  return 0;
}
