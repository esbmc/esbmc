#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define USAGE "./wronglock <param1> <param2>\n"

static int iNum1 = 2;
static int iNum2 = 1;
static int dataValue = 0;
pthread_mutex_t dataLock;
pthread_mutex_t thisLock;

void *funcA(void *param) {
	pthread_mutex_lock(&dataLock);
    int x = dataValue;
    dataValue++;
    if (dataValue != (x+1)) {
		assert(0);
    }
	pthread_mutex_unlock(&dataLock);

    return NULL;
}

void *funcB(void *param) {
	pthread_mutex_lock(&thisLock);
    dataValue++;
	pthread_mutex_unlock(&thisLock);

    return NULL;
}

int main(void) {
    int i;

	pthread_mutex_init(&dataLock, NULL);
	pthread_mutex_init(&thisLock, NULL);

    pthread_t num1Pool[iNum1];
    pthread_t num2Pool[iNum2];

    for (i = 0; i < iNum1; i++) {
		pthread_create(&num1Pool[i], NULL, &funcA, NULL);
    }

    for (i = 0; i < iNum2; i++) {
		pthread_create(&num2Pool[i], NULL, &funcB, NULL);
    }

    for (i = 0; i < iNum1; i++) {
		pthread_join(num1Pool[i], NULL);
    }

    for (i = 0; i < iNum2; i++) {
		pthread_join(num2Pool[i], NULL);
    }

    return 0;
}

