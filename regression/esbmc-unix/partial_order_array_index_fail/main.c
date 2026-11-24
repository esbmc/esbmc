/* #10990: t2 may read a[1][1] after t1 writes it. */
#include <pthread.h>
#include <assert.h>
int a[2][2];
unsigned nondet_uint();
void *t1(void *arg)
{
  a[1][1] = 5;
  return 0;
}
void *t2(void *arg)
{
  unsigned i = nondet_uint() % 2, j = nondet_uint() % 2;
  int r = a[i][j];
  assert(r == 0);
  return 0;
}
int main()
{
  pthread_t x, y;
  pthread_create(&x, 0, t1, 0);
  pthread_create(&y, 0, t2, 0);
  return 0;
}
