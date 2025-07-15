#include <pthread.h>
#include <assert.h>

pthread_rwlock_t lock;
int shared_counter = 0;

void *writer_thread(void *arg)
{
    int expected = *(int *)arg;
    pthread_rwlock_wrlock(&lock);
    shared_counter++;
    assert(shared_counter == expected);
    pthread_rwlock_unlock(&lock);
    return NULL;
}

int main()
{
    pthread_rwlock_init(&lock, NULL);

    pthread_t t1, t2;
    int expect1 = 1;
    int expect2 = 2;

    pthread_create(&t1, NULL, writer_thread, &expect1);
    pthread_join(t1, NULL);

    pthread_create(&t2, NULL, writer_thread, &expect2);
    pthread_join(t2, NULL);

    pthread_rwlock_wrlock(&lock);
    shared_counter++;
    assert(shared_counter == 3);
    pthread_rwlock_unlock(&lock);

    return 0;
}
