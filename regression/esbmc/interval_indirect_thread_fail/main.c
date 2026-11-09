#include <pthread.h>
int g;
void *w(void *p)
{
  g = 2;
  __ESBMC_assert(g == 2, "main may write g");
  return 0;
}
int main()
{
  pthread_t t;
  pthread_create(&t, 0, w, 0);
  g = 5;
  pthread_join(t, 0);
  return 0;
}
