#include <assert.h>
#include <pthread.h>

int x = 0;

void *writer(void *arg)
{
  x = 1;
  return 0;
}

void *checker(void *arg)
{
  assert(x == 0);
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, writer, 0);
  pthread_create(&b, 0, checker, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
