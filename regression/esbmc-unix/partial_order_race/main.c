/* #10990: the mutex keeps the read and the write of g apart. */
#include <pthread.h>
int g;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *reader(void *arg)
{
  pthread_mutex_lock(&m);
  int l = g;
  pthread_mutex_unlock(&m);
  return 0;
}
void *writer(void *arg)
{
  pthread_mutex_lock(&m);
  g = 1;
  pthread_mutex_unlock(&m);
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, reader, 0);
  pthread_create(&b, 0, writer, 0);
  return 0;
}
