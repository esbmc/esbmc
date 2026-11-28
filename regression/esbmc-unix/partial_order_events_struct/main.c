/* #10990: a field write updates the struct without a read of its own, and
   a field read reads the struct. */
#include <pthread.h>
struct s
{
  int a;
  int b;
} g;
void *t(void *arg)
{
  g.a = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  int b = g.b;
  return b;
}
