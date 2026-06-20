#include <pthread.h>

// An exception escaping a thread's start routine is uncaught for that thread and
// calls std::terminate ([except.terminate]). The lowering treats each function
// passed to pthread_create as a thread entry, so the escape is caught at the
// worker's own epilogue -> VERIFICATION FAILED. (Confirms removing the
// concurrency fallback did not drop uncaught-in-thread detection.)
struct E
{
};

void *worker(void *)
{
  throw E(); // not caught in the thread -> std::terminate
  return 0;
}

int main()
{
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  return 0;
}
