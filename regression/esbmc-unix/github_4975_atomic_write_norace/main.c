#include <pthread.h>

/* Companion to github_4975_atomic_write_race: the race-free twin.
 *
 * Here *every* access to the shared global `mThread` happens inside a
 * __VERIFIER_atomic_ region. Two atomic regions are mutually exclusive, so
 * there is no data race and the expected verdict is VERIFICATION SUCCESSFUL.
 *
 * This guards the #4975 fix against over-approximation: setting the write flag
 * for in-region writes must NOT add an assertion *inside* atomic regions, or
 * two mutually-exclusive atomic accesses to the same object would be reported
 * as a spurious race (a completeness regression).
 */

int mThread = 0;

void __VERIFIER_atomic_publish(int v)
{
  mThread = v; /* atomic write */
}

void __VERIFIER_atomic_read(int *out)
{
  *out = mThread; /* atomic read -- no race with the atomic write */
}

void *thr1(void *arg)
{
  __VERIFIER_atomic_publish(1);
  return 0;
}

void *thr2(void *arg)
{
  int self;
  __VERIFIER_atomic_read(&self);
  (void)self;
  return 0;
}

int main()
{
  pthread_t t;
  pthread_create(&t, 0, thr1, 0);
  thr2(0);
  return 0;
}
