
#include <pthread.h>
//#include <stdlib.h>
//#include <unistd.h>

/* Naive dining philosophers with inconsistent lock acquisition
   ordering. */

#define N 3
static pthread_t phil[N];
static pthread_mutex_t chop[N+1];

void* dine(void* arg)
{
   int i, *aptr1;
   aptr1 = (int *)arg;
   int left = *aptr1;
   int right = (left + 1) % N;
   assert(left>=0);
   assert(left<N);
   assert(right>=0);
   assert(right<N);

   for (i = 0; i < 1; i++) {
      pthread_mutex_lock(&chop[left]);
      pthread_mutex_lock(&chop[right]);
      /* eating */
      pthread_mutex_unlock(&chop[left]);
      pthread_mutex_unlock(&chop[right]);
   }
   return NULL;

}

int main ( void )
{
   int i, a;
   for (i = 0; i < N; i++)
   {
     pthread_mutex_init( &chop[i], NULL);
   }

   for (i = 0; i < N; i++)
   {
     a=i;
     pthread_create(&phil[i], NULL, dine, &a);
   }

//   sleep(1);

//   for (i = 0; i < N; i++)
//      pthread_join(phil[i], NULL);

   return 0;
}
