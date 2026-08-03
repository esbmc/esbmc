#include <pthread.h>

void *malloc(unsigned size);
pthread_mutex_t mutex;

#if 1
void *thread1(void *arg)
{
  pthread_mutex_lock(&mutex); 
  pthread_mutex_unlock(&mutex);
}
#endif

#if 1
void *thread2(void *arg)
{
  pthread_mutex_lock(&mutex); 
//  pthread_mutex_unlock(&mutex);
}
#endif

int main(void)
{
  pthread_t id1, id2;

//  mutex = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(&mutex, NULL);
//  assert(0);
  pthread_create(&id1, NULL, thread1, NULL);
  pthread_create(&id2, NULL, thread2, NULL);
  /* Returning from main calls exit() and tears the workers down before they
     can deadlock; pthread_exit keeps them alive (#6479). */
  pthread_exit(0);
}
