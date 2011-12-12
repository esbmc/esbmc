#include <pthread.h>
#include <assert.h>

#define N 2
#define FOOD 1

pthread_mutex_t  x[N];
pthread_mutex_t food_lock;

int food_on_table()
{
  static int food = FOOD;
  int myfood;
    	
  pthread_mutex_lock (&food_lock);
  if (food > 0) {
    food--;
  }
  myfood = food;
  pthread_mutex_unlock (&food_lock);
  return myfood;
}

void *thread1(void *arg)
{
  int id, *aptr1, left, right;
  
  aptr1=(int *)arg;
  id=*aptr1;

  left=id;
  right=(id+1)%N;
//  while(food_on_table())
//  {
  pthread_mutex_lock(&x[right]);
//  pthread_mutex_lock(&x[left]);
  pthread_mutex_unlock(&x[left]);
  pthread_mutex_unlock(&x[right]);
//  }
}

int main()
{
  int arg,i;
  pthread_t trd_id[N];

    pthread_mutex_init(&food_lock, NULL);

  for(i=0; i<N; i++)
    pthread_mutex_init(&x[i], NULL);

  for(i=0; i<N; i++)
  {
    arg=i;
    pthread_create(&trd_id[i], 0, thread1, &arg);
  }

  for(i=0; i<N; i++)
    pthread_join(trd_id[i], 0);

  return 0;
}

