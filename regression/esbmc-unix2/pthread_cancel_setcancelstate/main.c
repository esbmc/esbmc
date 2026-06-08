/* Verify that pthread_setcancelstate correctly saves and restores state. */
#include <pthread.h>
#include <assert.h>

int main()
{
  int old;

  /* Initial state must be ENABLE (0). */
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old);
  assert(old == PTHREAD_CANCEL_ENABLE);

  /* Restore and verify the round-trip. */
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old);
  assert(old == PTHREAD_CANCEL_DISABLE);

  /* Verify setcanceltype round-trip: initial type is DEFERRED (0). */
  int old_type;
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &old_type);
  assert(old_type == PTHREAD_CANCEL_DEFERRED);

  pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, &old_type);
  assert(old_type == PTHREAD_CANCEL_ASYNCHRONOUS);

  return 0;
}
