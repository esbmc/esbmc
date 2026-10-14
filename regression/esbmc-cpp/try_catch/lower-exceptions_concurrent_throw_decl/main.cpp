#include <new>
#include <pthread.h>

// A concurrent program that *declares* an exception specification but never
// throws or catches. <new>'s std::bad_alloc carries `what() const throw()`,
// whose no-throw spec emits a THROW_DECL — the same inert construct every CUDA
// program pulls in (call_kernel.h includes <new>), which used to make the
// concurrency gate reject the whole program. The exception-state tuple is shared
// across threads, so a concurrent program that genuinely throws/catches is
// reported as unsupported; but a bare exception specification puts nothing in
// flight, so the lowering must instead strip the THROW_DECL and verify normally
// (symex no longer has a THROW_DECL handler — a surviving one would abort). This
// pins that "concurrent + THROW_DECL-only" path to SUCCESSFUL.

int shared = 0;

void *worker(void *)
{
  shared++;
  return 0;
}

int main()
{
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  pthread_join(t, 0);
  return 0;
}
