#include <pthread.h>
#include <assert.h>
struct s
{
  int *p;
} g;
void *t(void *arg)
{
  g.p = 0;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(g.p == 0);
  return 0;
}
