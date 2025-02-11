#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>

#define USAGE "./reorder <param1> <param2>\n"

static int iSet = 1;
static int iCheck = 1;

static int a = 0;
static int b = 0;

void __ESBMC_yield();

void *setThread(void *param);
void *checkThread(void *param);
void set();
int check();

int main(int argc, char *argv[]) {
    int i, err;

    if (argc != 1) {
        if (argc != 3) {
            fprintf(stderr, USAGE);
            exit(-1);
        } else {
            sscanf(argv[1], "%d", &iSet);
            sscanf(argv[2], "%d", &iCheck);
        }
    }

    //printf("iSet = %d\niCheck = %d\n", iSet, iCheck);

    pthread_t setPool[iSet];
    pthread_t checkPool[iCheck];

    for (i = 0; i < iSet; i++) {
        if (0 != (err = pthread_create(&setPool[i], NULL, &setThread, NULL))) {
            fprintf(stderr, "Error [%d] found creating set thread.\n", err);
            exit(-1);
        }
    }

    for (i = 0; i < iCheck; i++) {
        if (0 != (err = pthread_create(&checkPool[i], NULL, &checkThread,
                                       NULL))) {
            fprintf(stderr, "Error [%d] found creating check thread.\n", err);
            exit(-1);
        }
    }

    for (i = 0; i < iSet; i++) {
        if (0 != (err = pthread_join(setPool[i], NULL))) {
            fprintf(stderr, "pthread join error: %d\n", err);
            exit(-1);
        }
    }

    for (i = 0; i < iCheck; i++) {
        if (0 != (err = pthread_join(checkPool[i], NULL))) {
            fprintf(stderr, "pthread join error: %d\n", err);
            exit(-1);
        }
    }

    return 0;
}
        
void *setThread(void *param) {
    a = 1;
    b = -1; /*BAD: W/W data race */

    return NULL;
}

void *checkThread(void *param) {
    if (! ((a == 0 && b == 0) || (a == 1 && b == -1))) {
        fprintf(stderr, "Bug found!\n");
        assert(0);
       // exit(-1);
    }

    return NULL;
}

