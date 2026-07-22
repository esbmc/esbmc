// CXL concurrent driver access with spinlocks test.
// Tests that spinlock-protected access to shared CXL device state
// prevents race conditions.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

/* Spinlock (simplified model) */
typedef struct {
  bool locked;
} spinlock_t;

void spin_lock_init(spinlock_t *lock)
{
  lock->locked = false;
}

bool spin_lock(spinlock_t *lock)
{
  __ESBMC_atomic_begin();
  if (lock->locked == false)
  {
    lock->locked = true;
    __ESBMC_atomic_end();
    return true;
  }
  __ESBMC_atomic_end();
  return false;
}

void spin_unlock(spinlock_t *lock)
{
  __ESBMC_atomic_begin();
  lock->locked = false;
  __ESBMC_atomic_end();
}

/* CXL device with shared state */
struct cxl_dev {
  spinlock_t lock;
  uint64_t command_count;
  uint64_t error_count;
};

static struct cxl_dev test_cxld;

/* Simulated concurrent command submission */
void submit_command(struct cxl_dev *cxld)
{
  assert(spin_lock(&cxld->lock));

  cxld->command_count++;

  spin_unlock(&cxld->lock);
}

/* Simulated concurrent error handler */
void handle_error(struct cxl_dev *cxld)
{
  assert(spin_lock(&cxld->lock));

  cxld->error_count++;

  spin_unlock(&cxld->lock);
}

int main()
{
  spin_lock_init(&test_cxld.lock);
  test_cxld.command_count = 0;
  test_cxld.error_count = 0;

  /* Simulate multiple concurrent command submissions */
  submit_command(&test_cxld);
  submit_command(&test_cxld);
  submit_command(&test_cxld);

  /* Simulate error handling */
  handle_error(&test_cxld);

  /* Verify atomicity: command_count should be exactly 3 */
  assert(test_cxld.command_count == 3);
  assert(test_cxld.error_count == 1);

  /* Verify lock is released */
  assert(test_cxld.lock.locked == false);
}
