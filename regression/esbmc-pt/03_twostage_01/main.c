#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>

#define USAGE "./twostage <param1> <param2>\n"

static int iTThreads = 1;
static int iRThreads = 1;
static int data1Value = 0;
static int data2Value = 0;
pthread_mutex_t *data1Lock;
pthread_mutex_t *data2Lock;

void lock(pthread_mutex_t *);
void unlock(pthread_mutex_t *);

void *funcA(void *param) {
    pthread_mutex_lock(data1Lock);
    data1Value = 1;
    pthread_mutex_unlock(data1Lock);

    pthread_mutex_lock(data2Lock);
    data2Value = data1Value + 1;
    pthread_mutex_unlock(data2Lock);

    return NULL;
}

void *funcB(void *param) {
    int t1 = -1;
    int t2 = -1;

    pthread_mutex_lock(data1Lock);
    if (data1Value == 0) {
        pthread_mutex_unlock(data1Lock);
        return NULL;
    }
    t1 = data1Value;
    pthread_mutex_unlock(data1Lock);

    pthread_mutex_lock(data2Lock);
    t2 = data2Value;
    pthread_mutex_unlock(data2Lock);

    if (t2 != (t1 + 1)) {
        fprintf(stderr, "Bug found!\n");
	assert(0); /* BAD */
        //exit(-1);
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    int i,err;

    if (argc != 1) {
        if (argc != 3) {
            fprintf(stderr, USAGE);
            exit(-1);
        } else {
            sscanf(argv[1], "%d", &iTThreads);
            sscanf(argv[2], "%d", &iRThreads);
        }
    }

    data1Lock = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
    data2Lock = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
    if (0 != (err = pthread_mutex_init(data1Lock, NULL))) {
        fprintf(stderr, "pthread_mutex_init error: %d\n", err);
        exit(-1);
    }
    if (0 != (err = pthread_mutex_init(data2Lock, NULL))) {
        fprintf(stderr, "pthread_mutex_init error: %d\n", err);
        exit(-1);
    }

    pthread_t tPool[iTThreads];
    pthread_t rPool[iRThreads];

    for (i = 0; i < iTThreads; i++) {
        if (0 != (err = pthread_create(&tPool[i], NULL, &funcA, NULL))) {
            fprintf(stderr, "Error [%d] found creating 2stage thread.\n", err);
            exit(-1);
        }
    }

    for (i = 0; i < iRThreads; i++) {
        if (0 != (err = pthread_create(&rPool[i], NULL, &funcB, NULL))) {
            fprintf(stderr, "Error [%d] found creating read thread.\n", err);
            exit(-1);
        }
    }

    for (i = 0; i < iTThreads; i++) {
        if (0 != (err = pthread_join(tPool[i], NULL))) {
            fprintf(stderr, "pthread join error: %d\n", err);
            exit(-1);
        }
    }

    for (i = 0; i < iRThreads; i++) {
        if (0 != (err = pthread_join(rPool[i], NULL))) {
            fprintf(stderr, "pthread join error: %d\n", err);
            exit(-1);
        }
    }

    return 0;
}

void lock(pthread_mutex_t *lock) {
    int err;
    if (0 != (err = pthread_mutex_lock(lock))) {
        fprintf(stderr, "Got error %d from pthread_mutex_lock.\n", err);
        exit(-1);
    }
}

void unlock(pthread_mutex_t *lock) {
    int err;
    if (0 != (err = pthread_mutex_unlock(lock))) {
        fprintf(stderr, "Got error %d from pthread_mutex_unlock.\n", err);
        exit(-1);
    }
}
   
