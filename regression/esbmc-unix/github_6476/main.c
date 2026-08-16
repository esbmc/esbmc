#include <pthread.h>

int main(void)
{
  pthread_mutex_t m;
  pthread_mutex_init(&m, 0);
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  pthread_mutex_destroy(&m);
  return 0;
}
