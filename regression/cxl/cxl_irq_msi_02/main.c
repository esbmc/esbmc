// CXL MSI teardown that frees the handler's context before unregistering the
// handler.
// Expected: VERIFICATION FAILED (driver bug: use-after-free in IRQ context,
// CWE-416)

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/irq.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/gfp.h>

#define CXL_IRQ_VECTORS 4

struct cxl_irq_ctx
{
  unsigned int irq;
  unsigned long events;
};

static void cxl_isr(int irq, void *dev_id)
{
  struct cxl_irq_ctx *ctx = (struct cxl_irq_ctx *)dev_id;
  (void)irq;
  ctx->events++;
}

int main()
{
  struct pci_dev dev;
  struct cxl_irq_ctx *ctx;
  int nvec;

  dev.vendor = 0x8086;
  dev.device = 0x0d93;

  ctx = kmalloc(sizeof(*ctx), GFP_KERNEL);
  if (!ctx)
    return 0;
  ctx->events = 0;

  nvec = pci_alloc_irq_vectors(&dev, 1, CXL_IRQ_VECTORS, 0);
  if (nvec < 0)
  {
    kfree(ctx);
    return 0;
  }

  if (pci_enable_msi(&dev))
  {
    pci_free_irq_vectors(&dev);
    kfree(ctx);
    return 0;
  }
  unsigned int irq = dev.irq;
  ctx->irq = irq;

  if (request_irq(irq, cxl_isr, 0, "cxl_msi", ctx))
  {
    pci_disable_msi(&dev);
    pci_free_irq_vectors(&dev);
    kfree(ctx);
    return 0;
  }

  unmask_irq(irq);
  esbmc_simulate_irq(irq, ctx);

  /*
   * BUG: disable_irq() stops delivery, so freeing here looks safe -- but the
   * handler is still registered, and pci_disable_msi() below re-enables the
   * line. Ordering teardown by what stops interrupts *now* rather than by what
   * makes the handler unreachable is the whole mistake; only free_irq()
   * guarantees the handler can never run again.
   */
  disable_irq(irq);
  kfree(ctx);

  enable_irq(irq);
  esbmc_simulate_irq(irq, ctx);

  free_irq(irq, ctx);
  pci_disable_msi(&dev);
  pci_free_irq_vectors(&dev);
  return 0;
}
