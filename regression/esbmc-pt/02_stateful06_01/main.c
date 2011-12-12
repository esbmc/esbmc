#include <pthread.h>
#include <assert.h>

#define NUM_ITE  3

pthread_mutex_t  ma;
int data;


void * thread1(void * arg)
{  
  int local1=0, local2=0;
  int i;

  for (i = 0; i < NUM_ITE; i++){  
    pthread_mutex_lock(&ma);
    data+=5;
    pthread_mutex_unlock(&ma);
  }
}


void * thread2(void * arg)
{  
  int local3 = 0, local4 = 0;
  int j;

  for (j = 0; j < NUM_ITE; j++){  
    pthread_mutex_lock(&ma);

    data += j;
    //  printf("thread2:%d: data = %d\n", j, data);
    assert(data % 5 == 2);
    pthread_mutex_unlock(&ma);
    
  }
}



int main()
{
  pthread_t  t_cap;
  pthread_t  t1, t2, t3;

  pthread_mutex_init(&ma, 0);

  data = 10;

  pthread_create(&t1, 0, thread1, 0);
  pthread_create(&t2, 0, thread2, 0);
  
  pthread_join(t1, 0);
  pthread_join(t2, 0);


  return 0;
}


