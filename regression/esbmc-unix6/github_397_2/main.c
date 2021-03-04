#include <pthread.h>
pthread_t c;
int d;
void *e(void *)
{
  d = 6;
}
int main()
{
  if(nondet_bool())
    pthread_create(&c, 0, e, 0);
  d = 3;
  assert(d == 3);
  return 0;
}
