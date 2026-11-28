/* #10990: a write through a pointer to a global struct, or to one of its
 * fields, updates the struct, which main reads after the join. */
#include <pthread.h>
#include <assert.h>
struct s
{
  int a;
  int v[2];
} g = {0, {3, 4}};
void *t(void *arg)
{
  struct s *p = &g;
  int *q = &g.a;
  *q = 1;
  p->v[1] = p->v[0] + 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  assert(g.a == 1 && g.v[0] == 3 && g.v[1] == 4);
  return 0;
}
