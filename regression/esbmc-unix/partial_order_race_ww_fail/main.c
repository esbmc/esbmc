/* #10990: two writes of g from different threads race. */
#include <pthread.h>
int g;
void *inside(void *arg)
{
  __ESBMC_atomic_begin();
  g = 1;
  __ESBMC_atomic_end();
  return 0;
}
void *outside(void *arg)
{
  g = 2;
  return 0;
}
void *also_outside(void *arg)
{
  g = 3;
  return 0;
}
int main()
{
  pthread_t a, b, c;
  pthread_create(&a, 0, inside, 0);
  pthread_create(&b, 0, outside, 0);
  pthread_create(&c, 0, also_outside, 0);
  return 0;
}
