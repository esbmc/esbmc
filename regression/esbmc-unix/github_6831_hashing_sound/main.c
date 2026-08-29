#include <assert.h>
#include <pthread.h>

int x = 0;

void *bump(void *arg)
{
  x++;
  x++;
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, bump, 0);
  pthread_create(&b, 0, bump, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x == 4);
  return 0;
}
