/* The kernel's spin_lock() blocks and has no failure path (esbmc/esbmc#1332),
   so holding the lock past the model's old SPIN_LIMIT must not lose thread 2's
   increment. This program was kernel_race_condition_spinlock_fail, which
   expected FAILED precisely because the old model gave up after SPIN_LIMIT
   retries -- behaviour the kernel does not have. */
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/spinlock.h>
#include <assert.h>
#include <pthread.h>
int shared_counter = 0;
spinlock_t lock;
#define MAX_THREADS 2

void* exceed_spin_limit(void* arg) {
   if(spin_lock(&lock))
   {
    for(int i=0; i<(SPIN_LIMIT+30); i++) { } // hold the lock past the old retry bound
    shared_counter++;
    spin_unlock(&lock);
   }

    return NULL;
}

void* normal_thread(void* arg) {
   if(spin_lock(&lock))
   {
    shared_counter++;
    spin_unlock(&lock);
   }
    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    spin_lock_init(&lock);
    pthread_create(&thread1, NULL, exceed_spin_limit, NULL);
    pthread_create(&thread2, NULL, normal_thread, NULL);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    assert(shared_counter == MAX_THREADS);
    return 0;
}