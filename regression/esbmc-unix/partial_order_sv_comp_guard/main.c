/* --sv-comp turns on --smt-symex-guard, which under --partial-order decided
 * each thread's busy-wait guard against the other thread's writes in program
 * order, so the check on Peterson's lock was taken and reported broken. */
#include <pthread.h>
#include <assert.h>
int flag1, flag2, turn, x;
void *t1(void *arg)
{
  flag1 = 1;
  turn = 1;
  while (flag2 == 1 && turn == 1)
    ;
  x = 0;
  if (x > 0)
    assert(0);
  flag1 = 0;
  return 0;
}
void *t2(void *arg)
{
  flag2 = 1;
  turn = 0;
  while (flag1 == 1 && turn == 0)
    ;
  x = 1;
  if (x < 1)
    assert(0);
  flag2 = 0;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
