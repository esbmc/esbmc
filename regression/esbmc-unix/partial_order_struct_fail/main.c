/* #10990: main may read g.a before the thread writes it and g.b after. */
#include <pthread.h>
#include <assert.h>
struct s
{
  int a;
  int b;
  int c;
} g = {0, 5, 7};
void *t(void *arg)
{
  g.a = 1;
  g.b = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  int a = g.a;
  int b = g.b;
  assert(b != 1 || a == 1);
  pthread_join(id, 0);
  return 0;
}
