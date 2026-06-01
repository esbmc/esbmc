// Regression test for github.com/esbmc/esbmc/issues/4434.
//
// The C `__thread` (GCC) / `_Thread_local` (C11) storage class gives
// each thread its own instance of the variable, initialized to the
// static initializer at thread start. Before the fix, ESBMC's frontend
// set symbolt::is_thread_local from clang::VarDecl::getTLSKind() but
// the symex layer routed every access through the shared
// level1_global SSA chain, so a write in one thread was visible to
// another. The pthread-race-challenges/thread-local-value SV-COMP
// benchmark reported a spurious unreach-call violation because
// thread 2's first read of `data` resolved to thread 1's prior write.
//
// The fix gives `__thread` globals a per-thread level1 instance (see
// renaming.cpp::is_thread_local) and seeds the initial value from
// pthread_trampoline via __ESBMC_init_thread_local() so the first
// read in each spawned thread sees the static initializer (here, 0).
#include <pthread.h>
#include <assert.h>

__thread int data = 0;

void *worker(void *arg)
{
  assert(data == 0);
  data = 1;
  assert(data == 1);
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
