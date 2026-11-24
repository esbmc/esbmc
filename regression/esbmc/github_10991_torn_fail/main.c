#include <assert.h>
#include <pthread.h>
#include <string.h>

unsigned nondet_uint(void);

char d[4];
unsigned n;

void *writer(void *arg)
{
  memset(d, 1, n);
  return 0;
}

int main()
{
  n = nondet_uint();
  __ESBMC_assume(n >= 2 && n <= 4);
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  char a = d[0];
  char b = d[1];
  assert(!(a == 1 && b == 0));
  return 0;
}
