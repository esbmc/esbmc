// CXL mailbox submission without serialisation: two threads drive the one
// mailbox register set concurrently.
// Expected: VERIFICATION FAILED (driver bug: unsynchronised mailbox access)

#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

static struct cxl_dev cxld;
static int in_flight;

/*
 * BUG: drivers/cxl/core/mbox.c takes cxl_mailbox::mbox_mutex around the
 * whole submit-and-await sequence, because the device has one mailbox
 * register set. This driver takes no lock, so two submissions interleave:
 * in_flight is read-modify-written concurrently, and the second command
 * starts while the first is still outstanding.
 */
static void *submit(void *arg)
{
  (void)arg;

  in_flight++;
  assert(in_flight == 1);

  cxl_mailbox_ioctl(&cxld, CXL_MBOX_OP_GET_STATUS, NULL, 0);

  in_flight--;

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

  return 0;
}
