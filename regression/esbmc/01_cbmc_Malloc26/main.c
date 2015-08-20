#include <stdio.h>
#include <pthread.h>
//#include <stdlib.h>

//void *malloc(unsigned size);
//void free(void *p);

#define ACCTS 5

typedef struct Account {
    char name;
    double amount;
    pthread_mutex_t *lock;
} Account;

static Account *accounts[ACCTS];

Account *newAccount(char nm, double amt) {
    int err;

    Account *tmp = (Account *) malloc(sizeof(Account));
    __ESBMC_assume(tmp);
    tmp->lock = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
    __ESBMC_assume(tmp->lock);
    tmp->name = nm;
    tmp->amount = amt;
    if (0 != (err = pthread_mutex_init(tmp->lock, NULL))) {
        fprintf(stderr, "Got error %d from pthread_mutex_init.\n", err);
        exit(-1);
    }
    return tmp;
}

int main()
{
	int i, err;
	char names[ACCTS] = {'A','B','C','D','E'};
    for (i = 0; i < ACCTS; i++) {
        accounts[i] = (Account *) malloc(sizeof(Account));
        __ESBMC_assume(accounts[i]);
        accounts[i] = newAccount(names[i], 100);
    }

    accounts[0]->name='c';
    assert(accounts[0]->name!='c');
}
