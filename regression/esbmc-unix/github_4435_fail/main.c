// Negative companion to github_4435: same two-writer state-hashing setup,
// but an assertion that genuinely fails (both threads run, so seen[1]
// cannot be 0). Confirms ESBMC still finds the legitimate counterexample
// under --state-hashing with the new (thename, level1_num, thread_num)
// hash key; guards against the fix accidentally suppressing real bugs.

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
  assert(seen[0] == 1 && seen[1] == 0);
  return 0;
}
