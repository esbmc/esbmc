// CXL driver IRQ handler verification test.
// Tests that the IRQ handler correctly processes CXL events:
// mailbox completion, error events, and port events.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>

typedef int irqreturn_t;
#define IRQ_HANDLED 1
#define IRQ_NONE 0

/* CXL interrupt types */
enum cxl_irq_type {
  CXL_IRQ_MAILBOX = 0,
  CXL_IRQ_ERROR,
  CXL_IRQ_PORT,
};

struct cxl_dev {
  void *regs;
  enum cxl_irq_type last_irq;
  uint32_t mailbox_opcode;
  uint32_t error_code;
  int irq_count;
  int mailbox_count;
  int error_count;
  int port_count;
};

static struct cxl_dev test_cxld;

/* IRQ handler: dispatch based on interrupt type */
irqreturn_t cxl_irq_handler(int irq, void *dev_id)
{
  (void)irq;
  struct cxl_dev *cxld = (struct cxl_dev *)dev_id;
  assert(cxld != NULL);

  cxld->irq_count++;

  /* Determine interrupt type from hardware register */
  enum cxl_irq_type type = (enum cxl_irq_type)__VERIFIER_nondet_int() % 3;
  cxld->last_irq = type;

  switch (type)
  {
  case CXL_IRQ_MAILBOX:
    /* Read mailbox status register */
    cxld->mailbox_opcode = __VERIFIER_nondet_uint();
    cxld->mailbox_count++;
    break;

  case CXL_IRQ_ERROR:
    /* Read error status register */
    cxld->error_code = __VERIFIER_nondet_uint();
    cxld->error_count++;
    break;

  case CXL_IRQ_PORT:
    /* Port event: enumerate new devices */
    cxld->port_count++;
    break;
  }

  return IRQ_HANDLED;
}

int main()
{
  test_cxld.regs = (void *)0xFED00000;
  test_cxld.last_irq = CXL_IRQ_MAILBOX;
  test_cxld.irq_count = 0;
  test_cxld.mailbox_count = 0;
  test_cxld.error_count = 0;
  test_cxld.port_count = 0;

  /* Simulate multiple IRQ firings */
  for (int i = 0; i < 10; i++)
  {
    irqreturn_t ret = cxl_irq_handler(42, &test_cxld);
    assert(ret == IRQ_HANDLED);
  }

  /* Verify counts */
  assert(test_cxld.irq_count == 10);
  assert(test_cxld.mailbox_count + test_cxld.error_count + test_cxld.port_count == 10);

  /* Verify last IRQ type is valid */
  assert(test_cxld.last_irq >= CXL_IRQ_MAILBOX && test_cxld.last_irq <= CXL_IRQ_PORT);
}
