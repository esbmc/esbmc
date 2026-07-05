// A pthread_create whose start routine is passed through a function-pointer
// variable (a computed routine) — collect_thread_entry cannot resolve it to a
// single function. The lowering over-approximates by treating every address-
// taken void*(void*) function as a thread entry, so the uncaught-escape check
// still applies: worker throws E and never catches it, so the exception escapes
// the thread's initial function and std::terminate is called ([except.terminate])
// — VERIFICATION FAILED.
#include <pthread.h>

struct E
{
};

void *worker(void *)
{
  throw E();
  return 0;
}

int main()
{
  pthread_t t;
  void *(*fp)(void *) = worker; // computed: not a direct &worker argument
  pthread_create(&t, 0, fp, 0);
  pthread_join(t, 0);
  return 0;
}
