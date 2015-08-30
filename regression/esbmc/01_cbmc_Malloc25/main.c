#include <stdio.h>
//#include <stdlib.h>

//void *malloc(unsigned size);
//void free(void *p);

#define ACCTS 5

typedef struct Account {
    int name;
    double amount;
} Account;

static Account *accounts[ACCTS];

int main()
{
  accounts[0] = (Account *) malloc(sizeof(Account));
  __ESBMC_assume(accounts[0]);
  accounts[0]->name=1;
  assert(accounts[0]->name!=1);
}
