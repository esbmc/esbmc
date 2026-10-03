#include <pthread.h>
#include <assert.h>

/* Companion to state_hashing_cswitch_sound_fail: with only 2 context
 * switches the buggy interleaving is genuinely unreachable, so the result
 * is SUCCESSFUL. This confirms state-hashing still prunes soundly (the fix
 * does not simply disable pruning). */

int a;

void *W0(void *arg)
{
  a = 1;
  if (a == 0)
    assert(0);
  return 0;
}

void *W1(void *arg)
{
  a = 0;
  a = 2;
  return 0;
}

int main()
{
  pthread_t t0, t1;
  pthread_create(&t0, 0, W0, 0);
  pthread_create(&t1, 0, W1, 0);
  return 0;
}
