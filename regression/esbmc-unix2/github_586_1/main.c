#include <pthread.h>
int turn; // integer variable to hold the ID of the thread whose turn is it
int x;    // variable to test mutual exclusion
int flag1 = 0, flag2 = 0; // boolean flags

void thr1()
{ // frontend produces 12 transitions from this thread. It would be better if it would produce only 8!
  flag1 = 1;
  turn = 1;
  do
  {
  } while (flag2 == 1 && turn == 1);
  // begin: critical section
  x = 0;
  assert(x <= 0);
  // end: critical section
  flag1 = 0;
}

void thr2()
{
  flag2 = 1;
  turn = 0;
  do
  {
  } while (flag1 == 1 && turn == 0);
  // begin: critical section
  x = 1;
  assert(x >= 1);
  // end: critical section
  flag2 = 0;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}
