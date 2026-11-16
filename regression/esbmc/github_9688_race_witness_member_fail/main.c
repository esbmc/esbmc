#include <pthread.h>
struct S
{
  int a[3];
} s;
struct S *p = &s;
int r;
void *t(void *arg)
{
  p->a[1] = 1;
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  r = s.a[1];
  pthread_join(id, 0);
  return 0;
}
