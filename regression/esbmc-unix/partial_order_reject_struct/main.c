#include <pthread.h>
#include <assert.h>
struct s
{
  int f;
} g;
void *t(void *arg)
{
  g.f = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(g.f == 0);
  return 0;
}
