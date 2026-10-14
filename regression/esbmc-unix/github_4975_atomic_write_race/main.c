#include <pthread.h>

/* Regression for esbmc/esbmc#4975 (pthread-ext/23_lu-fig2.fixed).
 *
 * A R/W data race on the global `mThread`:
 *   - thr1 writes `mThread` *inside* a __VERIFIER_atomic_ region;
 *   - thr2 reads `mThread` with no synchronisation (its first statement).
 *
 * A __VERIFIER_atomic_ region only serialises a thread against *other* atomic
 * regions; it establishes no happens-before relation with thr2's plain read,
 * so the atomic write and the unsynchronised read are concurrent conflicting
 * accesses -- a data race. The sv-benchmarks ground truth for this task is
 * `no-data-race: false`.
 *
 * Before the fix, add_race_assertions emitted the in-region write as a bare
 * assignment and never set its write flag, so thr2's RACE_CHECK(&mThread)
 * could never observe a concurrent writer: ESBMC reported VERIFICATION
 * SUCCESSFUL on a racy program (a soundness / false-negative bug).
 */

int mThread = 0;
int start_main = 0;

void __VERIFIER_atomic_publish(int v)
{
  mThread = v; /* shared write performed inside an atomic region */
}

void *thr1(void *arg)
{
  start_main = 1;
  __VERIFIER_atomic_publish(1);
  return 0;
}

void *thr2(void *arg)
{
  int self = mThread; /* unsynchronised read -- races with thr1's atomic write */
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
