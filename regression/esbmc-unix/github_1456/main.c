#include <pthread.h>

int a[4];

int index1 = 0;
int index2 = 1;

void* t1(void *arg) {
    a[index1] = 1;    // a[0]             

    return NULL;
}

void* t2(void *arg) {
    a[index2] = 1;    // a[1]

    return NULL;
}

int main() {
    pthread_t id1, id2;

    pthread_create(&id1, NULL, t1, NULL);
    pthread_create(&id2, NULL, t2, NULL);

    return 0;
}
