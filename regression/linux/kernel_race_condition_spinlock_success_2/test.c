#include <ubuntu20.04/kernel_5.15.0-76/include/linux/spinlock.h>
#include <assert.h>
#include <pthread.h>
int shared_counter = 0;
spinlock_t lock;
#define MAX_THREADS 3

void *lockAndIncrement(void *arg) {
    spin_lock(&lock);
    shared_counter++;
    spin_unlock(&lock);
}

int main() {
    pthread_t threads[MAX_THREADS];
    spin_lock_init(&lock);
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_create(&threads[i], NULL, lockAndIncrement, NULL);
    }
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    assert(shared_counter == MAX_THREADS);
    return 0;
}