/* #10990: a struct is one object, so main sees the thread's field writes in
 * program order, and the field the thread does not write keeps its value. */
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
  int b = g.b;
  int a = g.a;
  assert(b != 1 || a == 1);
  pthread_join(id, 0);
  assert(g.a == 1 && g.b == 1 && g.c == 7);
  return 0;
}
