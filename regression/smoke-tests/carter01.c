#include <pthread.h>
#include <stdio.h>

pthread_mutex_t  mutex, lock;
int A_count = 0, B_count = 0;


void * thread_A(void* arg)
{
  pthread_mutex_lock(&mutex);
  A_count++;
  if (A_count == 1) 
    pthread_mutex_lock(&lock);
  pthread_mutex_unlock(&mutex);

  pthread_mutex_lock(&mutex);
  A_count--;
  if (A_count == 0) 
    pthread_mutex_unlock(&lock);
  pthread_mutex_unlock(&mutex);
}


void * thread_B(void * arg)
{
  pthread_mutex_lock(&mutex);
  B_count++;
  if (B_count == 1) 
    pthread_mutex_lock(&lock);
  pthread_mutex_unlock(&mutex);

  pthread_mutex_lock(&mutex);
  B_count--;
  if (B_count == 0) 
    pthread_mutex_unlock(&lock);
  pthread_mutex_unlock(&mutex);

}



int main()
{

  pthread_mutex_init(&mutex,NULL);
  pthread_mutex_init(&lock,NULL);

  pthread_t a1, a2, b1, b2;

  pthread_create(&a1, NULL, thread_A, NULL);
  
  pthread_create(&b1, NULL, thread_B, NULL);

  pthread_join(a1, NULL);
  pthread_join(b1, NULL);

  return 0;
}
