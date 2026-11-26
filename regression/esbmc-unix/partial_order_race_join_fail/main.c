/* #10990: main writes g before joining the thread that writes it. */
#include <pthread.h>
int g;
void *t(void *arg)
{
  g = 1;
  return 0;
}
int main()
{
  pthread_t a;
  g = 0;
  pthread_create(&a, 0, t, 0);
  g = 2;
  pthread_join(a, 0);
  return 0;
}
