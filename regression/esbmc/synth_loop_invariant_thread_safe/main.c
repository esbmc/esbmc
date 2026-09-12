/* The other half of the thread decline: a threaded program whose claim holds
 * keeps its SUCCESSFUL verdict. Declining synthesis costs the proof nothing
 * here -- BMC already decides the claim -- so the guard only removes the
 * shape synth_loop_invariant_thread_falseproof pins, not working proofs. */
#include <pthread.h>

unsigned int g;

void *writer(void *arg)
{
  unsigned int i = 0;
  g = 0;
  while (i < 3)
  {
    g = g + 1;
    i = i + 1;
  }
  return 0;
}

void *reader(void *arg)
{
  __ESBMC_assert(g <= 3, "g <= 3");
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, writer, 0);
  pthread_create(&t2, 0, reader, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  return 0;
}
