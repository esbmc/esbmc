/* #10990: a write on a branch never taken does not race. */
#include <pthread.h>
int g;
int flag;
void *reader(void *arg)
{
  int l = g;
  return 0;
}
void *writer(void *arg)
{
  if (flag)
    g = 1;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, reader, 0);
  pthread_create(&b, 0, writer, 0);
  return 0;
}
