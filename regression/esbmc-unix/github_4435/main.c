// Smoke test for PR #4444 (refs #4435): under --state-hashing, two pthread
// workers that each declare a local variable with the same C name (here
// `tid`) must not collapse into the same state-hash bucket. Before the
// fix, current_hashes was keyed on `thename` alone, so the two threads'
// `tid` overwrote each other's per-symbol hash. This positive test does
// not by itself reproduce the false alarm reported in #4435 (which needs
// SV-COMP-scale state explosion), but exercises the new key path and
// guards against trivial regressions of the state-hashing machinery on
// multi-threaded programs with same-named thread-local variables.

#include <pthread.h>
#include <assert.h>

int seen[2] = {0, 0};

void *writer(void *arg)
{
  int tid = *(int *)arg;
  seen[tid] = 1;
  return 0;
}

int main(void)
{
  pthread_t t0, t1;
  int a = 0, b = 1;
  pthread_create(&t0, 0, writer, &a);
  pthread_create(&t1, 0, writer, &b);
  pthread_join(t0, 0);
  pthread_join(t1, 0);
  assert(seen[0] == 1 && seen[1] == 1);
  return 0;
}
