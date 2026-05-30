#include <pthread.h>

extern void __VERIFIER_atomic_begin(void);
extern void __VERIFIER_atomic_end(void);
extern void __VERIFIER_assume(int);

/* No-race companion to github_4423_atomic_race (esbmc/esbmc#4423).
 *
 * Identical control flow, but the accesses to `stopped` are synchronized
 * (wrapped in atomic regions). There is therefore no data race, and the result
 * must stay VERIFICATION SUCCESSFUL. This guards the #4423 fix against the
 * opposite failure mode: the extra atomic-boundary interleaving points it adds
 * must not fabricate a spurious race on a correctly synchronized program.
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
  __VERIFIER_atomic_begin();
  stopped = 1; /* synchronized write */
  __VERIFIER_atomic_end();
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
    int observed;
    __VERIFIER_atomic_begin();
    observed = stopped; /* synchronized read */
    __VERIFIER_atomic_end();
    if (observed)
      pendingIo = pendingIo;
  }
  BCSP_IoDecrement();
  return 0;
}
