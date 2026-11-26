/* #10990: a write on one branch still races with the other thread's read. */
#include <pthread.h>
int g;
int nondet_int();
void *reader(void *arg)
{
  int l = g;
  return 0;
}
void *writer(void *arg)
{
  if (nondet_int())
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
