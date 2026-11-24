/* #10990: under --partial-order main runs to completion first, then each
   spawned thread in creation order, and every shared write is an event. */
#include <pthread.h>
int x;
void *t1(void *a)
{
  x = 1;
  return 0;
}
void *t2(void *a)
{
  x = 2;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  x = 3;
  return 0;
}
