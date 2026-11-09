// #9867: an _Atomic pointer does not make its pointee atomic, so *p races.
#include <pthread.h>
int v;
int *_Atomic p = &v;

void *thr(void *arg)
{
  *p = 1;
  return 0;
}

int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, thr, 0);
  *p = 2;
  pthread_join(id, 0);
  return 0;
}
