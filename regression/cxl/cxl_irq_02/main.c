// CXL IRQ double-free verification test.
// Tests that free_irq called twice does not corrupt state.
// Expected: VERIFICATION FAILED (double-free bug)

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#define IRQ_HANDLED 1

typedef void (*irq_handler_t)(int irq, void *dev_id);

struct cxl_dev {
  void *regs;
  int irq_registered;
  int free_count; /* tracks how many times free_irq was called */
};

static struct cxl_dev test_cxld;

int request_irq(unsigned int irq, irq_handler_t handler, unsigned long flags,
                const char *name, void *dev_id)
{
  (void)irq; (void)handler; (void)flags; (void)name;
  struct cxl_dev *cxld = (struct cxl_dev *)dev_id;
  cxld->irq_registered = 1;
  cxld->free_count = 0;
  return 0;
}

void free_irq(unsigned int irq, void *dev_id)
{
  (void)irq;
  struct cxl_dev *cxld = (struct cxl_dev *)dev_id;
  cxld->free_count++;
  /* BUG: No check if already freed — allows double-free */
  cxld->irq_registered = 0;
}

int main()
{
  test_cxld.regs = (void *)0x1000;
  test_cxld.irq_registered = 0;
  test_cxld.free_count = 0;

  /* Register */
  request_irq(42, NULL, 0, "cxl-test", &test_cxld);
  assert(test_cxld.irq_registered == 1);

  /* First free — OK */
  free_irq(42, &test_cxld);
  assert(test_cxld.irq_registered == 0);
  assert(test_cxld.free_count == 1);

  /*
   * BUG: Second free — should be rejected but isn't.
   * In a real driver this could corrupt kernel state.
   */
  free_irq(42, &test_cxld);

  /*
   * The invariant: free_irq should only be called once per request_irq.
   * free_count should never exceed 1.
   */
  __ESBMC_assert(test_cxld.free_count <= 1,
                 "free_irq called more than once (double-free)");
}
