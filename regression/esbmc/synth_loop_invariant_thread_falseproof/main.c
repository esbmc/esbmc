/* The reason synthesis declines on a threaded program.
 *
 * Cutting the writer's loop replaces its four visits with a havoc and one body
 * execution, so g takes at most two distinct values in the cut program. The
 * reader's claim needs three, and a violation that is unreachable is an UNSAT
 * claim -- which the #7491 classifier reports PASSED, because it acts on the
 * refutation side only. Without the decline this run was VERIFICATION
 * SUCCESSFUL under --no-vacuity-check on a program BMC reports FAILED. */
#include <pthread.h>

unsigned int g;

void *writer(void *arg)
{
  unsigned int i = 0;
  g = 0;
  while (i < 4)
  {
    g = g + 1;
    i = i + 1;
  }
  return 0;
}

void *reader(void *arg)
{
  unsigned int a = g;
  unsigned int b = g;
  unsigned int c = g;
  __ESBMC_assert(!(a == 1 && b == 2 && c == 3), "three distinct observations");
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
