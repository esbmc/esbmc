#include <pthread.h>

/* Sleep sets must not trust a subtree that --no-unwinding-assertions truncated
 * (issue #6831). The unwind bound cuts this loop with an assumption, which
 * drives the state guard false; get_next_formula's interleaving_unviable break
 * then abandons the pending switch. That break is deliberately not a
 * truncation -- for a genuinely infeasible guard it loses nothing -- but here
 * the remaining iterations are merely unexplored, so a thread slept against
 * the frame and the W/R race on x was reported SUCCESSFUL.
 *
 * Reduced from esbmc-unix2/11_podelski.fig3.lics04 (Podelski & Rybalchenko,
 * LICS'04 fig. 3), which is where the flip was first observed.
 */

int x = 1;
int y = 0;

void *reader(void *arg)
{
  (void)arg;
  while (x == 1)
    y = y + 1;
  return 0;
}

void *writer(void *arg)
{
  (void)arg;
  x = 0;
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, reader, 0);
  pthread_create(&t2, 0, writer, 0);
  return 0;
}
