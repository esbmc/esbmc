/* Atomic cast in an if-condition: one thread writes to an atomic_uint
 * (CK_NonAtomicToAtomic), another reads it as an if-condition
 * (CK_AtomicToNonAtomic).  Neither access should be flagged as a race. */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_uint flag = 0;

void *writer(void *arg)
{
  flag = 1; /* CK_NonAtomicToAtomic */
  return NULL;
}

void *checker(void *arg)
{
  if (flag) /* CK_AtomicToNonAtomic in condition */
    flag = 2; /* CK_NonAtomicToAtomic in unbraced body */
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, writer, NULL);
  pthread_create(&t2, NULL, checker, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  assert(flag >= 1);
  return 0;
}
