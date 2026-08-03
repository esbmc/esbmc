// CXL MSI interrupt setup and teardown in the correct order: the handler is
// unregistered before the state it touches is freed.
// Expected: VERIFICATION SUCCESSFUL

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
  ctx->irq = dev.irq;

  if (request_irq(ctx->irq, cxl_isr, 0, "cxl_msi", ctx))
  {
    pci_disable_msi(&dev);
    pci_free_irq_vectors(&dev);
    kfree(ctx);
    return 0;
  }

  unmask_irq(ctx->irq);
  esbmc_simulate_irq(ctx->irq, ctx);
  assert(ctx->events == 1);

  /* Quiesce, then unregister, then free. synchronize_irq() waits for an
     in-flight handler; free_irq() guarantees none can start afterwards. Only
     then is ctx unreachable from interrupt context. */
  disable_irq(ctx->irq);
  synchronize_irq(ctx->irq);
  mask_irq(ctx->irq);
  free_irq(ctx->irq, ctx);

  /* No handler is registered now, so this delivery must find nothing. */
  esbmc_simulate_irq(ctx->irq, ctx);
  assert(ctx->events == 1);

  enable_irq(ctx->irq);
  pci_disable_msi(&dev);
  pci_free_irq_vectors(&dev);
  kfree(ctx);
  return 0;
}
