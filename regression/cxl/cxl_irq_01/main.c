// CXL IRQ registration and firing verification test.
// Tests that interrupt handlers are called correctly when IRQs fire.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* IRQ types */
#define IRQ_HANDLED 1
#define IRQ_NONE    0

/* Handler type */
typedef void (*irq_handler_t)(int irq, void *dev_id);

/* Simulated CXL device */
struct cxl_dev {
  void *regs;
  int irq_number;
  irq_handler_t handler;
  void *handler_dev_id;
  int irq_registered;
  int irq_fired_count;
};

static struct cxl_dev test_cxld;

/* Simulated IRQ handler */
void cxl_irq_handler(int irq, void *dev_id)
{
  struct cxl_dev *cxld = (struct cxl_dev *)dev_id;
  assert(cxld != NULL);
  assert(irq == cxld->irq_number);
  cxld->irq_fired_count++;
}

/* Simulated request_irq */
int request_irq(unsigned int irq, irq_handler_t handler, unsigned long flags,
                const char *name, void *dev_id)
{
  (void)flags; (void)name;
  assert(handler != NULL);
  assert(dev_id != NULL);
  struct cxl_dev *cxld = (struct cxl_dev *)dev_id;
  cxld->irq_number = (int)irq;
  cxld->handler = handler;
  cxld->handler_dev_id = dev_id;
  cxld->irq_registered = 1;
  return 0;
}

/* Simulated free_irq */
void free_irq(unsigned int irq, void *dev_id)
{
  (void)irq;
  struct cxl_dev *cxld = (struct cxl_dev *)dev_id;
  cxld->irq_registered = 0;
  cxld->handler = NULL;
}

/* Simulate IRQ firing */
void simulate_irq(unsigned int irq, void *dev_id)
{
  struct cxl_dev *cxld = (struct cxl_dev *)dev_id;
  if (cxld->handler != NULL && cxld->irq_registered)
  {
    cxld->handler((int)irq, dev_id);
  }
}

int main()
{
  test_cxld.regs = (void *)0x1000;
  test_cxld.irq_registered = 0;
  test_cxld.irq_fired_count = 0;

  /* Register IRQ handler */
  int ret = request_irq(42, cxl_irq_handler, 0, "cxl-test", &test_cxld);
  assert(ret == 0);
  assert(test_cxld.irq_registered == 1);

  /* Simulate IRQ firing */
  simulate_irq(42, &test_cxld);
  assert(test_cxld.irq_fired_count == 1);

  /* Simulate another IRQ */
  simulate_irq(42, &test_cxld);
  assert(test_cxld.irq_fired_count == 2);

  /* Unregister */
  free_irq(42, &test_cxld);
  assert(test_cxld.irq_registered == 0);

  /* IRQ should not fire after free */
  simulate_irq(42, &test_cxld);
  assert(test_cxld.irq_fired_count == 2); /* unchanged */
}
