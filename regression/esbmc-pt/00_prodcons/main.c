#include <pthread.h>
#include <stdio.h>

#define AMOUNT_PROD 5

#define AMOUNT_CONS 1

#define CS_OVERHEAD 1
#define DEADLINE 45
#define TRUE  1
#define FALSE 0

//@ DEFINE-MAIN-TIMER timer
unsigned int timer = 0;
void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();
int __ESBMC_activeThread = -1;
_Bool join[AMOUNT_PROD+AMOUNT_CONS];

int item = 0;

pthread_mutex_t  lock = PTHREAD_MUTEX_INITIALIZER;
int thread_id[AMOUNT_PROD+AMOUNT_CONS];

void *producer(void *v) 
{
  //@ WCET-BLOCK [8]
  __ESBMC_atomic_begin();
  timer += 8; 
  int id = *(int*) v;
  pthread_mutex_lock(&lock);
  item = id;
  printf("Producer (%d): item = %d.\n", id, item);
  pthread_mutex_unlock(&lock);
  if (__ESBMC_activeThread != id) timer += CS_OVERHEAD; 
  __ESBMC_activeThread = id;
  //@ ASSERT-TIMER (timer <= DEADLINE)
  assert (timer <= DEADLINE);
  join[id] = TRUE;
  __ESBMC_atomic_end();
  //@ BLOCK END

  return NULL;
}

void *consumer(void *v) 
{
  //@ WCET-BLOCK [7]
  __ESBMC_atomic_begin();
  timer += 7;
  int id = *(int*) v;
  pthread_mutex_lock(&lock);
  printf("Consumer (%d): item = %d.\n", id, item);
  pthread_mutex_unlock(&lock);
  if (__ESBMC_activeThread != id) timer += CS_OVERHEAD; 
  __ESBMC_activeThread = id;
  //@ ASSERT-TIMER (timer <= DEADLINE)
  assert (timer <= DEADLINE);
  join[id] = TRUE;
  __ESBMC_atomic_end();
  //@ BLOCK END

  return NULL;
}


int main() 
{

  pthread_t thr_prod[AMOUNT_PROD];
  pthread_t thr_cons[AMOUNT_CONS];
  int i, j;

  __ESBMC_atomic_begin();
  for (i=0; i<(AMOUNT_PROD+AMOUNT_CONS); i++)
    join[i] = FALSE;
    
  // Creating PRODUCER'S threads
  for (i = 0; i < AMOUNT_PROD ; i++) {
    thread_id[i] = i;
    pthread_create(&thr_prod[i], NULL, producer, &thread_id[i]);
  }
  
  i = AMOUNT_PROD;
  // Creating CONSUMER'S threads
  for (j = 0; j < AMOUNT_CONS; j++) {
    thread_id[i] = i;
    pthread_create(&thr_cons[j], NULL, consumer, &thread_id[i]);
    i++;
  }

  __ESBMC_atomic_end();

  pthread_mutex_destroy (&lock);

  return 0;
}

