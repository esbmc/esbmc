/* Simple possible deadlock */
#include <pthread.h>

pthread_mutex_t m1;// = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m2;// = PTHREAD_MUTEX_INITIALIZER;

void __ESBMC_yield();

static void *t1(void *v)
{
  pthread_mutex_lock(&m1); //lock 1
  pthread_mutex_lock(&m2); //lock 2
//  pthread_mutex_unlock(&m1);
//  pthread_mutex_unlock(&m2);

  return NULL;
}

static void *t2(void *v)
{
  pthread_mutex_lock(&m1); //lock 4
  pthread_mutex_lock(&m2); //lock 3
//  pthread_mutex_unlock(&m1);
//  pthread_mutex_unlock(&m2);

  return NULL;
}

int main()
{
  pthread_t a, b;
	
  pthread_mutex_init(&m1, NULL);
  pthread_mutex_init(&m2, NULL);

  pthread_create(&a, NULL, t1, NULL);	
  pthread_create(&b, NULL, t2, NULL);

//  pthread_join(a, NULL);
//  pthread_join(b, NULL);

  return 0;
}

