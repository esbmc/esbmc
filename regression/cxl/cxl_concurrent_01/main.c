// CXL concurrent driver access with spinlocks.
// Two submitters and an error handler contend for the same device state.
// The spinlock must serialise them, so no update is lost and no access races.
// Expected: VERIFICATION SUCCESSFUL

#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

/* Spinlock (simplified model).
 *
 * Acquisition is atomic and *blocking*: the assume discards the schedules in
 * which the lock is already held, which is what a caller spinning on it would
 * observe. Modelling it as a try-lock instead — returning false under
 * contention — is only sound when nothing contends, which is precisely the
 * case this test exists to exclude. */
typedef struct
{
  bool locked;
} spinlock_t;

void spin_lock_init(spinlock_t *lock)
{
  lock->locked = false;
}

void spin_lock(spinlock_t *lock)
{
  __ESBMC_atomic_begin();
  __ESBMC_assume(lock->locked == false);
  lock->locked = true;
  __ESBMC_atomic_end();
}

void spin_unlock(spinlock_t *lock)
{
  __ESBMC_atomic_begin();
  lock->locked = false;
  __ESBMC_atomic_end();
}

/* CXL device with shared state */
struct cxl_dev
{
  spinlock_t lock;
  uint64_t command_count;
  uint64_t error_count;
};

static struct cxl_dev test_cxld;

void submit_command(struct cxl_dev *cxld)
{
  spin_lock(&cxld->lock);
  cxld->command_count++;
  spin_unlock(&cxld->lock);
}

void handle_error(struct cxl_dev *cxld)
{
  spin_lock(&cxld->lock);
  cxld->error_count++;
  spin_unlock(&cxld->lock);
}

static void *submitter(void *arg)
{
  (void)arg;
  submit_command(&test_cxld);
  return NULL;
}

static void *errorer(void *arg)
{
  (void)arg;
  handle_error(&test_cxld);
  return NULL;
}

int main()
{
  pthread_t s1;
  pthread_t s2;
  pthread_t e1;

  spin_lock_init(&test_cxld.lock);
  test_cxld.command_count = 0;
  test_cxld.error_count = 0;

  pthread_create(&s1, NULL, submitter, NULL);
  pthread_create(&s2, NULL, submitter, NULL);
  pthread_create(&e1, NULL, errorer, NULL);
  pthread_join(s1, NULL);
  pthread_join(s2, NULL);
  pthread_join(e1, NULL);

  /* Neither increment was lost to a concurrent read-modify-write. */
  assert(test_cxld.command_count == 2);
  assert(test_cxld.error_count == 1);
  assert(test_cxld.lock.locked == false);

  return 0;
}
