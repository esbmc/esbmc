#include <pthread.h>

void* t1(void *arg) {
    int* shared = (int*) arg;
    *shared = 4;                 //race here

    return NULL;
}

void* t2(void *arg) {
    int* shared = (int*) arg;
    *shared = 2;                 //race here

    return NULL;
}

int main() {
    pthread_t id1, id2;
    int shared_data = 0;

    pthread_create(&id1, NULL, t1, &shared_data);
    pthread_create(&id2, NULL, t2, &shared_data);

    pthread_join(id1, NULL);
    pthread_join(id2, NULL);

    return 0;
}
