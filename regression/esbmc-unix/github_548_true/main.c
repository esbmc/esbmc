#include <assert.h>
extern void abort(void);
void reach_error() { assert(0); }
void __VERIFIER_assert(int cond) { if(!(cond)) { ERROR: {reach_error();abort();} } }

#include<pthread.h>

int glob1 = 0;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg) {
  pthread_mutex_lock(&mutex2);
  glob1 = 5;

  pthread_mutex_unlock(&mutex2);
  return NULL;
}

void *foo(void *arg)
{
  pthread_mutex_lock(&mutex2); // wait/block mutex2
  __VERIFIER_assert(glob1 == 0 || glob1 == 5); // 
  pthread_mutex_unlock(&mutex2); // unlock mutex2
}

int main(void) {
  pthread_t id, id2;  
  pthread_create(&id, NULL, t_fun, NULL);
  *(foo)(0);
  pthread_join (id, NULL);  
  return 0;
}