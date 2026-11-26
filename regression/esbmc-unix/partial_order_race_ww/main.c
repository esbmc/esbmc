/* #10990: writes of g inside atomic sections do not race. */
#include <pthread.h>
int g;
void *inside(void *arg)
{
  __ESBMC_atomic_begin();
  g = 1;
  __ESBMC_atomic_end();
  return 0;
}
void *also_inside(void *arg)
{
  __ESBMC_atomic_begin();
  g = 2;
  __ESBMC_atomic_end();
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, inside, 0);
  pthread_create(&b, 0, also_inside, 0);
  return 0;
}
