#include <ubuntu20.04/kernel_5.15.0-76/include/linux/spinlock.h>
#include <assert.h>
#include <pthread.h>
int shared_counter = 0;
spinlock_t lock;
#define MAX_THREADS 2

void * increment_counter(void *)
{
    //apply spin lock
    spin_lock(&lock);
    //increment counter
    shared_counter++;
    spin_unlock(&lock);
}

int main()
{
    pthread_t threads[MAX_THREADS];

    spin_lock_init(&lock);

    for (int i = 0; i < MAX_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, increment_counter, NULL);

    }
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    //check if post condition is satisfied
    assert(shared_counter == MAX_THREADS);
    return 0;
}
