/* Dekker's entry protocol: each thread sets its flag and waits while the
 * other's is set. Symex runs thr1 first, and its last write clears flag1;
 * thr2's loop guard must still read flag1 from the clocks, not from thr1's
 * final value. */
#include <pthread.h>
#include <assert.h>
int flag1, flag2, x, turn;
void *thr1(void *a)
{
  flag1 = 1;
  while (flag2 >= 1)
  {
    if (turn != 0)
    {
      flag1 = 0;
      flag1 = 1;
    }
  }
  x = 0;
  assert(x <= 0);
  flag1 = 0;
  return 0;
}
void *thr2(void *a)
{
  flag2 = 1;
  while (flag1 >= 1)
  {
    if (turn != 1)
    {
      flag2 = 0;
      flag2 = 1;
    }
  }
  x = 1;
  flag2 = 0;
  return 0;
}
int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, thr1, 0);
  pthread_create(&t2, 0, thr2, 0);
  return 0;
}
