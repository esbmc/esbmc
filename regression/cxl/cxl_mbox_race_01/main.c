// CXL mailbox submission under concurrency: the device has one mailbox
// register set, so the driver serialises submission. With the lock held the
// single-command-in-flight invariant holds and no access races.
// Expected: VERIFICATION SUCCESSFUL

#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

static struct cxl_dev cxld;

/* drivers/cxl/core/mbox.c serialises on cxl_mailbox::mbox_mutex. */
static pthread_mutex_t mbox_mutex = PTHREAD_MUTEX_INITIALIZER;
static int in_flight;

static void *submit(void *arg)
{
  (void)arg;

  pthread_mutex_lock(&mbox_mutex);

  in_flight++;
  /* The hardware exposes a single mailbox: two commands must never be
     outstanding at once. */
  assert(in_flight == 1);

  cxl_mailbox_ioctl(&cxld, CXL_MBOX_OP_GET_STATUS, NULL, 0);

  in_flight--;
  pthread_mutex_unlock(&mbox_mutex);

  return NULL;
}

int main()
{
  pthread_t a;
  pthread_t b;

  pthread_create(&a, NULL, submit, NULL);
  pthread_create(&b, NULL, submit, NULL);
  pthread_join(a, NULL);
  pthread_join(b, NULL);

  /* Every command that started also finished. */
  assert(in_flight == 0);

  return 0;
}
