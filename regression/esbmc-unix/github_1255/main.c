#include <pthread.h>

void* t1(void *arg) {
    int* shared = (int*) arg;
    *shared = 2;                 // race here

    return NULL;
}

void* t2(void *arg) {
    int* shared = (int*) arg;
    *shared = 3;                 

    return NULL;
}

int main() {
    pthread_t id1, id2;
    int shared_data;

    pthread_create(&id1, NULL, t1, &shared_data);
    pthread_create(&id2, NULL, t2, &shared_data);

    return 0;
}
