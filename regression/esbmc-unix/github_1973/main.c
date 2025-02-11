#include <pthread.h>
#include <assert.h>

pthread_t t1, t2;

_Bool flag = 0;

void *thread2(void *arg) {
   // part 1
   assert(!flag);
   return 0;
}

void *thread1(void *arg) {
    pthread_create(&t2, NULL, thread2, NULL);
    return 0;
}

int main(void) {
    if (nondet_int()) 
    {
      pthread_create(&t1, NULL, thread1, NULL);
      return 0;
    }
    // part 2
    flag = 1;

    return 0;
}
