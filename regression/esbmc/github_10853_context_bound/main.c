#include <pthread.h>
int g = 0, flag = 0;
void *t(void *arg)
{
  int a = flag;
  if (a == 1 && flag == 2)
    g = 1;
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  flag = 1;
  flag = 2;
  pthread_join(id, 0);
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 3 && g == 0);
  while (i < 3)
    i++;
  return 0;
}
