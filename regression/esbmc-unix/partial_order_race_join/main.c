/* #10990: spawn and join order main's writes of g around the thread's. */
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
  pthread_join(a, 0);
  g = 2;
  return 0;
}
