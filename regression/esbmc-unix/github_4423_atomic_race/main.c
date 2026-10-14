#include <pthread.h>

extern void __VERIFIER_atomic_begin(void);
extern void __VERIFIER_atomic_end(void);
extern void __VERIFIER_assume(int);

/* Regression for esbmc/esbmc#4423 (pthread-lit/qw2004-2).
 *
 * A W/R data race on `stopped` that can only be exposed by interleaving
 * *between* two atomic regions of the increment path: the reader must observe
 * stoppingFlag == 0 (atomic region A) and then be suspended before touching
 * pendingIo (atomic region B/C) while the writer thread runs to completion and
 * performs the non-atomic write `stopped = 1`.
 *
 * In --data-races-check-only mode all interleaving points are derived from the
 * yield() calls add_race_assertions emits around non-atomic accesses, so before
 * the fix there was no context-switch point at the A->B atomic-region boundary
 * and the race was missed (VERIFICATION SUCCESSFUL on a racy program).
 */

volatile int stoppingFlag;
volatile int pendingIo;
volatile int stoppingEvent;
volatile int stopped;

int BCSP_IoIncrement(void)
{
  int lsf;
  __VERIFIER_atomic_begin();
  lsf = stoppingFlag; /* atomic region A: read shared flag */
  __VERIFIER_atomic_end();
  if (lsf)
    return -1;
  int lp;
  __VERIFIER_atomic_begin();
  lp = pendingIo; /* atomic region B: read shared counter */
  __VERIFIER_atomic_end();
  __VERIFIER_atomic_begin();
  pendingIo = lp + 1; /* atomic region C: write shared counter */
  __VERIFIER_atomic_end();
  return 0;
}

int dec(void)
{
  int tmp;
  __VERIFIER_atomic_begin();
  pendingIo = pendingIo - 1;
  tmp = pendingIo;
  __VERIFIER_atomic_end();
  return tmp;
}

void BCSP_IoDecrement(void)
{
  if (dec() == 0)
  {
    __VERIFIER_atomic_begin();
    stoppingEvent = 1;
    __VERIFIER_atomic_end();
  }
}

void *BCSP_PnpStop(void *arg)
{
  (void)arg;
  __VERIFIER_atomic_begin();
  stoppingFlag = 1;
  __VERIFIER_atomic_end();
  BCSP_IoDecrement();
  int lse;
  __VERIFIER_atomic_begin();
  lse = stoppingEvent;
  __VERIFIER_atomic_end();
  __VERIFIER_assume(lse);
  stopped = 1; /* NON-atomic write -> races with read below */
  return 0;
}

int main(void)
{
  pthread_t t;
  pendingIo = 1;
  stoppingFlag = 0;
  stoppingEvent = 0;
  stopped = 0;
  pthread_create(&t, 0, BCSP_PnpStop, 0);

  if (BCSP_IoIncrement() == 0)
  {
    int observed = stopped; /* NON-atomic read -> races with write above */
    if (observed)
      pendingIo = pendingIo;
  }
  BCSP_IoDecrement();
  return 0;
}
