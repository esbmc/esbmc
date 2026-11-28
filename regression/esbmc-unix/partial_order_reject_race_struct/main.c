/* #10990: a struct is one object, so its fields cannot be checked apart. */
#include <pthread.h>
struct s
{
  int a;
  int b;
} g;
void *t(void *arg)
{
  g.b = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  g.a = 1;
  return 0;
}
