// Positive control for lower-exceptions_unsupported_concurrency: a concurrent
// program that does NOT use exceptions still verifies normally. The lowering
// pass only rejects a concurrent program when it actually uses exceptions
// (report_unsupported's program_uses_exceptions guard), so for exception-free
// concurrency the pass is a silent no-op and verification proceeds (#5075).
#include <pthread.h>

int g = 0;

void *worker(void *)
{
  g = 1;
  return 0;
}

int main()
{
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  return 0;
}
