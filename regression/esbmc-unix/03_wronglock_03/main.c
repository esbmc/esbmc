#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>

#define USAGE "./wronglock <param1> <param2>\n"

static int iNum1 = 1;
static int iNum2 = 1;
static int dataValue = 0;
pthread_mutex_t *dataLock;
pthread_mutex_t *thisLock;

void lock(pthread_mutex_t *);
void unlock(pthread_mutex_t *);

void __ESBMC_yield();

void *funcA(void *param) {
    lock(dataLock);
    int x = dataValue;
    dataValue++;
	//__ESBMC_yield();
    if (dataValue != (x+1)) {
//        fprintf(stderr, "Bug Found!\n");
		assert(0);
//        exit(-1);
    }
    unlock(dataLock);

    return NULL;
}

void *funcB(void *param) {
    lock(thisLock);
    dataValue++;
    unlock(thisLock);

    return NULL;
}

int main(int argc, char *argv[]) {
    int i,err;

    if (argc != 1) {
        if (argc != 3) {
            fprintf(stderr, USAGE);
            exit(-1);
        } else {
            sscanf(argv[1], "%d", &iNum1);
            sscanf(argv[2], "%d", &iNum2);
        }
    }

    dataLock = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
    thisLock = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
    if (0 != (err = pthread_mutex_init(dataLock, NULL))) {
        fprintf(stderr, "pthread_mutex_init error: %d\n", err);
        exit(-1);
    }
    if (0 != (err = pthread_mutex_init(thisLock, NULL))) {
        fprintf(stderr, "pthread_mutex_init error: %d\n", err);
        exit(-1);
    }

    pthread_t num1Pool[iNum1];
    pthread_t num2Pool[iNum2];

    for (i = 0; i < iNum1; i++) {
        if (0 != (err = pthread_create(&num1Pool[i], NULL, &funcA, NULL))) {
            fprintf(stderr, "Error [%d] found creating num1 thread.\n", err);
            exit(-1);
        }
    }

    for (i = 0; i < iNum2; i++) {
        if (0 != (err = pthread_create(&num2Pool[i], NULL, &funcB, NULL))) {
            fprintf(stderr, "Error [%d] found creating num2 thread.\n", err);
            exit(-1);
        }
    }

    for (i = 0; i < iNum1; i++) {
        if (0 != (err = pthread_join(num1Pool[i], NULL))) {
            fprintf(stderr, "pthread join error: %d\n", err);
            exit(-1);
        }
    }

    for (i = 0; i < iNum2; i++) {
        if (0 != (err = pthread_join(num2Pool[i], NULL))) {
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
