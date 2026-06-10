// Negative companion to github_4431_atomic_ptr_norace: identical structure but
// the pointer targets a plain (non-atomic) int, so the two unsynchronised
// updates through the pointer ARE a genuine data race. This confirms that the
// atomic-element exclusion does NOT over-suppress real races on non-atomic
// objects reached through a pointer.
//
// Expected: VERIFICATION FAILED.
#include <pthread.h>

int storage;
int *A;

void *t_inc(void *arg)
{
  A[0]++; // RACE!
  return 0;
}
void *t_dec(void *arg)
{
  A[0]--; // RACE!
  return 0;
}

int main(void)
{
  A = &storage;
  pthread_t x, y;
  pthread_create(&x, 0, t_inc, 0);
  pthread_create(&y, 0, t_dec, 0);
  pthread_join(x, 0);
  pthread_join(y, 0);
  return 0;
}
