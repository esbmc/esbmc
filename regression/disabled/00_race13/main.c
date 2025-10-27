#include <pthread.h>
#include <assert.h>

int nondet_int();

pthread_mutex_t m, n;
int x, y, z, balance;
_Bool deposit_done=0, withdraw_done=0;

void *deposit(void *arg) 
{
  pthread_mutex_lock(&m); //lock 3
  balance = balance + y;
  deposit_done=1;
  pthread_mutex_unlock(&m);
}

void *withdraw(void *arg) 
{
  pthread_mutex_lock(&n); //lock 1
  balance = balance - z;
  withdraw_done=1;
  pthread_mutex_unlock(&n);
}

void *check_result(void *arg) 
{
  pthread_mutex_lock(&m); //lock 2
  pthread_mutex_lock(&n);
  if (deposit_done && withdraw_done)
    assert(balance == (x + y) - z);
  pthread_mutex_unlock(&m);
  pthread_mutex_unlock(&n);
}

int main() 
{
  pthread_t t1, t2, t3;

  pthread_mutex_init(&m, 0);
  pthread_mutex_init(&n, 0);

  x = nondet_int();
  y = nondet_int();
  z = nondet_int();
  balance = x;

  pthread_create(&t3, 0, check_result, 0);
  pthread_create(&t1, 0, deposit, 0);
  pthread_create(&t2, 0, withdraw, 0);

  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  pthread_join(t3, NULL);

  return 0;
}
