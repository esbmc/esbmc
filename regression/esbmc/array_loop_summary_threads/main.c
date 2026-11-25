#include <pthread.h>
int a[2];
void *t(void *arg)
{
  for (int i = 0; i < 2; i++)
    a[i] = 1;
  return 0;
}
int main()
{
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  pthread_join(th, 0);
  __ESBMC_assert(a[0] == 1 && a[1] == 1, "written");
  return 0;
}
