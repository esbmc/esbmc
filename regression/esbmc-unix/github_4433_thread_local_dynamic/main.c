// Regression test for github.com/esbmc/esbmc/issues/4433.
//
// Sibling of #4434: same `__thread` modelling but with a pointer-typed
// TLS variable and a per-thread dynamic allocation. Each thread starts
// with `data == NULL`, allocates its own buffer via calloc, asserts
// it is zero-initialized, writes to it, and frees. Without the fix
// (see renaming.cpp + intrinsic_init_thread_local), thread 2's first
// read of `data` resolved to thread 1's calloc'd pointer, breaking
// the assertion. The fix gives `__thread int *data = NULL` per-thread
// SSA storage seeded with NULL at thread start.
#include <pthread.h>
#include <stdlib.h>
#include <assert.h>

__thread int *data = 0;

void *worker(void *arg)
{
  data = (int *)calloc(2, sizeof(int));
  assert(data != 0);
  assert(data[0] == 0);
  data[0] = 1;
  assert(data[0] == 1);
  free(data);
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, worker, 0);
  pthread_create(&t2, 0, worker, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  return 0;
}
