#include <pthread.h>
#include <stdio.h>

int x = 0;

/* The read of x is an argument to printf, which the frontend lowers to an
 * OTHER instruction rather than a call -- so the read reaches MPOR's access
 * sets only via execution_statet::symex_printf. Without it the reader looks
 * independent of the writer and the sleep set never wakes. */
void *reader(void *arg)
{
  printf("%d\n", x);
  return 0;
}

void *writer(void *arg)
{
  x = 1;
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, reader, 0);
  pthread_create(&b, 0, writer, 0);
  return 0;
}
