#include <pthread.h>
#include <assert.h>

#define  N  3

int  num;
unsigned long total;
int flag;

pthread_mutex_t  m;
pthread_cond_t  empty, full;

void * thread1(void * arg)
{
  int i;

  i = 0;
  while (i < N){
    pthread_mutex_lock(&m);
    while (num > 0) 
      pthread_cond_wait(&empty, &m);
    
    num++;

    printf ("produce ....%d\n", i);
    pthread_mutex_unlock(&m);

    pthread_cond_signal(&full);

    i++;
  }
}


void * thread2(void * arg)
{
  int j;

  j = 0;
  while (j < N){
    pthread_mutex_lock(&m);
    while (num == 0) 
      pthread_cond_wait(&full, &m);

    total=total+j;
    printf("total ....%d\n",total);
    num--;
    printf("consume ....%d\n",j);
    pthread_mutex_unlock(&m);
    
    pthread_cond_signal(&empty);
    j++;    
  }
  total=total+j;
  printf("total ....%d\n",total);
  flag=1;
}


int main()
{
  pthread_t  t1, t2;

  num = 0;
  total = 0;

  pthread_mutex_init(&m, 0);
  pthread_cond_init(&empty, 0);
  pthread_cond_init(&full, 0);
  
  pthread_create(&t1, 0, thread1, 0);

  pthread_create(&t2, 0, thread2, 0);

  pthread_join(t1, 0);
  pthread_join(t2, 0);
  
  if (flag)
    assert(total==((N*(N+1))/2));
  return 0;
  
}

