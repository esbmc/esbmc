/*
 * cxl_driver.c — Operational models for CXL (Compute Express Link) device
 *                driver verification in ESBMC.
 *
 * These models abstract the behavior of CXL device drivers so that ESBMC can
 * verify driver code without a real CXL hardware backend.  Each function
 * returns a non-deterministic result consistent with the CXL 2.0/3.0
 * specification, along with appropriate precondition checks.
 *
 * Key design decisions:
 *   - MMIO space is modelled as a global byte array (ESBMC_MMIO_SPACE).
 *     readl/writel etc. read/write from/to this space at the device's
 *     mapped offset.  Writes store; a read returns the stored value once
 *     the driver has written that register, and a nondet value before then
 *     (unknown power-on hardware state).
 *   - DMA coherent memory is modelled as a separate global array
 *     (ESBMC_DMA_SPACE) so that device-visible data is distinct from
 *     kernel virtual memory.
 *   - PCI devices are modelled as a small array (ESBMC_PCI_DEVICES_MAX)
 *     with non-deterministic enumeration.
 *   - Interrupts are modelled as a callback array; esbmc_simulate_irq()
 *     invokes the registered handler with a non-deterministic IRQ number.
 *   - CXL mailbox commands return non-deterministic status codes,
 *     constrained by __ESBMC_assume() to valid CXL spec ranges.
 *
 * Reference: kernel.c for kernel API modeling, socket_lib.c for
 * non-deterministic interface modeling.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <assert.h>

/* Include kernel headers for types used in models */
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/gfp.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/spinlock.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/asm/io.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/irq.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/dma-mapping.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxlmem.h>

/* ============================================================
 *  Non-deterministic value generators (ESBMC built-ins)
 * ============================================================ */
extern int __VERIFIER_nondet_int(void);
extern uint8_t __VERIFIER_nondet_uchar(void);
extern uint16_t __VERIFIER_nondet_ushort(void);
extern uint32_t __VERIFIER_nondet_uint(void);
extern unsigned long __VERIFIER_nondet_ulong(void);
extern size_t __VERIFIER_nondet_size_t(void);

/* ============================================================
 *  Memory space declarations
 * ============================================================ */

/* MMIO space — models the device register address space.
 * 64 KB per device is generous but covers all CXL register blocks. */
#define ESBMC_MMIO_SPACE_SIZE (64 * 1024)
#define ESBMC_PCI_DEVICES_MAX 16
#define ESBMC_DMA_SPACE_SIZE (1024 * 1024) /* 1 MB DMA-coherent space */
#define ESBMC_IRQ_HANDLERS_MAX 32

static char esbmc_mmio_space[ESBMC_MMIO_SPACE_SIZE];
/* Per-byte shadow: 1 once the driver has written that byte of register space */
static char esbmc_mmio_written[ESBMC_MMIO_SPACE_SIZE];
static char esbmc_dma_space[ESBMC_DMA_SPACE_SIZE];
static int esbmc_dma_used[ESBMC_DMA_SPACE_SIZE]; /* tracking for assertions */

/* PCI device table.  Holds caller-supplied pointers rather than copies, so a
 * device found by pci_get_device() is the same object the harness set up and
 * per-device state (AER, driver_data) keyed on the pointer stays consistent. */
static struct pci_dev *esbmc_pci_devices[ESBMC_PCI_DEVICES_MAX];
static int esbmc_pci_count = 0;

/* IRQ handler table */
static struct
{
  unsigned int irq;
  irq_handler_t handler;
  void *dev_id;
} esbmc_irq_table[ESBMC_IRQ_HANDLERS_MAX];
static int esbmc_irq_count = 0;

/* ============================================================
 *  MMIO access models
 * ============================================================
 *
 * MMIO reads return non-deterministic values (the device may return
 * anything).  MMIO writes store to the MMIO space so that subsequent
 * reads of the same address reflect the written value — this models
 * read-back of writable registers.  Some registers are modelled as
 * RO (read-only) and ignore writes.
 * ============================================================ */

/* Resolve a pointer to an offset in the MMIO space, rejecting accesses whose
 * full width does not fit — checking only the first byte would let readq/writeq
 * near the top of the space run past esbmc_mmio_space.  Stays in pointer
 * arithmetic throughout: casting through uintptr_t loses the object
 * association in ESBMC's pointer model, which leaves the offset unconstrained
 * and makes every read take the non-deterministic path. */
static inline int
__esbmc_mmio_offset(const void *addr, size_t width, size_t *off)
{
__ESBMC_HIDE:;
  const char *p = (const char *)addr;
  if (
    p < esbmc_mmio_space ||
    p > esbmc_mmio_space + (ESBMC_MMIO_SPACE_SIZE - width))
    return 0;
  *off = (size_t)(p - esbmc_mmio_space);
  return 1;
}

/* A register byte reads back only once the driver has written it; until then
 * its content is whatever the hardware left there, i.e. non-deterministic. */
static inline int __esbmc_mmio_is_written(size_t off, size_t width)
{
__ESBMC_HIDE:;
  for (size_t i = 0; i < width; i++)
    if (!esbmc_mmio_written[off + i])
      return 0;
  return 1;
}

static inline void __esbmc_mmio_mark_written(size_t off, size_t width)
{
__ESBMC_HIDE:;
  for (size_t i = 0; i < width; i++)
    esbmc_mmio_written[off + i] = 1;
}

uint8_t readb(const void *addr)
{
__ESBMC_HIDE:;
  size_t off;
  if (!__esbmc_mmio_offset(addr, sizeof(uint8_t), &off))
    return 0; /* out-of-bounds read returns 0 */
  if (!__esbmc_mmio_is_written(off, sizeof(uint8_t)))
    return __VERIFIER_nondet_uchar();
  return *(volatile uint8_t *)((void *)addr);
}

uint16_t readw(const void *addr)
{
__ESBMC_HIDE:;
  size_t off;
  if (!__esbmc_mmio_offset(addr, sizeof(uint16_t), &off))
    return 0;
  if (!__esbmc_mmio_is_written(off, sizeof(uint16_t)))
    return __VERIFIER_nondet_ushort();
  return *(volatile uint16_t *)((void *)addr);
}

uint32_t readl(const void *addr)
{
__ESBMC_HIDE:;
  size_t off;
  if (!__esbmc_mmio_offset(addr, sizeof(uint32_t), &off))
    return 0;
  if (!__esbmc_mmio_is_written(off, sizeof(uint32_t)))
    return __VERIFIER_nondet_uint();
  return *(volatile uint32_t *)((void *)addr);
}

uint64_t readq(const void *addr)
{
__ESBMC_HIDE:;
  size_t off;
  if (!__esbmc_mmio_offset(addr, sizeof(uint64_t), &off))
    return 0;
  if (!__esbmc_mmio_is_written(off, sizeof(uint64_t)))
    return (uint64_t)__VERIFIER_nondet_ulong();
  return *(volatile uint64_t *)((void *)addr);
}

void writeb(uint8_t val, const void *addr)
{
__ESBMC_HIDE:;
  size_t off;
  if (!__esbmc_mmio_offset(addr, sizeof(val), &off))
    return;
  *(volatile uint8_t *)((void *)addr) = val;
  __esbmc_mmio_mark_written(off, sizeof(val));
}

void writew(uint16_t val, const void *addr)
{
__ESBMC_HIDE:;
  size_t off;
  if (!__esbmc_mmio_offset(addr, sizeof(val), &off))
    return;
  *(volatile uint16_t *)((void *)addr) = val;
  __esbmc_mmio_mark_written(off, sizeof(val));
}

void writel(uint32_t val, const void *addr)
{
__ESBMC_HIDE:;
  size_t off;
  if (!__esbmc_mmio_offset(addr, sizeof(val), &off))
    return;
  *(volatile uint32_t *)((void *)addr) = val;
  __esbmc_mmio_mark_written(off, sizeof(val));
}

void writeq(uint64_t val, const void *addr)
{
__ESBMC_HIDE:;
  size_t off;
  if (!__esbmc_mmio_offset(addr, sizeof(val), &off))
    return;
  *(volatile uint64_t *)((void *)addr) = val;
  __esbmc_mmio_mark_written(off, sizeof(val));
}

/* Relaxed variants — same as strict, no ordering guarantee */
uint32_t readl_relaxed(const void *addr)
{
  return readl(addr);
}

void writel_relaxed(uint32_t val, const void *addr)
{
  writel(val, addr);
}

/* Block operations — copy between MMIO and kernel memory */
void readsl(const void *addr, void *buf, unsigned long count)
{
__ESBMC_HIDE:;
  assert(addr != NULL);
  assert(buf != NULL);
  assert(count > 0);
  /* Each 32-bit word is non-deterministic */
  for (unsigned long i = 0; i < count; i++)
  {
    ((uint32_t *)buf)[i] = readl((const char *)addr + i * 4);
  }
}

void writesl(const void *addr, const void *buf, unsigned long count)
{
__ESBMC_HIDE:;
  assert(addr != NULL);
  assert(buf != NULL);
  assert(count > 0);
  for (unsigned long i = 0; i < count; i++)
  {
    writel(((const uint32_t *)buf)[i], (const char *)addr + i * 4);
  }
}

/* Memory barriers — no-op in the model (ordering is verified by ESBMC's
 * thread interleaving, not by hardware barriers). */
void mb(void)
{ /* no-op */
}
void wmb(void)
{ /* no-op */
}
void rmb(void)
{ /* no-op */
}
void smp_mb(void)
{ /* no-op */
}
void smp_wmb(void)
{ /* no-op */
}
void smp_rmb(void)
{ /* no-op */
}

/* ============================================================
 *  I/O port functions (stub — not used by CXL, but declared)
 * ============================================================ */
uint8_t inb(unsigned long port)
{
  (void)port;
  return __VERIFIER_nondet_uchar();
}
uint16_t inw(unsigned long port)
{
  (void)port;
  return __VERIFIER_nondet_ushort();
}
uint32_t inl(unsigned long port)
{
  (void)port;
  return __VERIFIER_nondet_uint();
}
void outb(uint8_t val, unsigned long port)
{
  (void)val;
  (void)port;
}
void outw(uint16_t val, unsigned long port)
{
  (void)val;
  (void)port;
}
void outl(uint32_t val, unsigned long port)
{
  (void)val;
  (void)port;
}

/* ============================================================
 *  PCI device models
 * ============================================================ */

int esbmc_pci_register_device(struct pci_dev *dev)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  if (esbmc_pci_count >= ESBMC_PCI_DEVICES_MAX)
    return -ENOSPC;
  esbmc_pci_devices[esbmc_pci_count++] = dev;
  return 0;
}

void esbmc_pci_reset_devices(void)
{
  esbmc_pci_count = 0;
}

static struct pci_dev *
__esbmc_pci_find_by_id(u16 vendor, u16 device, struct pci_dev *from)
{
  int start = 0;
  if (from != NULL)
  {
    /* Resume after 'from', per pci_get_device()'s iteration contract. A 'from'
     * that is not in the table ends the walk rather than restarting it. */
    start = esbmc_pci_count;
    for (int i = 0; i < esbmc_pci_count; i++)
    {
      if (esbmc_pci_devices[i] == from)
      {
        start = i + 1;
        break;
      }
    }
    if (start >= esbmc_pci_count)
      return NULL;
  }

  for (int i = start; i < esbmc_pci_count; i++)
  {
    if (
      (vendor == 0 || esbmc_pci_devices[i]->vendor == vendor) &&
      (device == 0 || esbmc_pci_devices[i]->device == device))
    {
      return esbmc_pci_devices[i];
    }
  }
  return NULL;
}

struct pci_dev *pci_get_device(u16 vendor, u16 device, struct pci_dev *from)
{
__ESBMC_HIDE:;
  return __esbmc_pci_find_by_id(vendor, device, from);
}

struct pci_dev *pci_get_bus_device(u32 domain, u8 bus, u8 devfn)
{
__ESBMC_HIDE:;
  (void)domain;
  (void)bus;
  (void)devfn;
  if (esbmc_pci_count == 0)
    return NULL;
  int idx = __VERIFIER_nondet_int();
  __ESBMC_assume(idx >= 0 && idx < esbmc_pci_count);
  return esbmc_pci_devices[idx];
}

void pci_put_device(struct pci_dev *dev)
{
  /* no-op in the model */
  (void)dev;
}

int pci_enable_device(struct pci_dev *dev)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = ENODEV;
    return -1;
  }
  return 0;
}

void pci_disable_device(struct pci_dev *dev)
{
  (void)dev;
}

int pci_request_regions(struct pci_dev *dev, const char *res_name)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(res_name != NULL);
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = EBUSY;
    return -1;
  }
  return 0;
}

void pci_release_regions(struct pci_dev *dev)
{
  (void)dev;
}

resource_size_t pci_resource_start(struct pci_dev *dev, int bar)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(bar >= 0 && bar < PCI_NUM_RESOURCES_WITH_ROM);
  /* Return a non-deterministic BAR address aligned to 4 KB */
  return (resource_size_t)(__VERIFIER_nondet_ulong() & ~0xFFFUL) &
         (resource_size_t)(~0xFFFUL);
}

resource_size_t pci_resource_end(struct pci_dev *dev, int bar)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(bar >= 0 && bar < PCI_NUM_RESOURCES_WITH_ROM);
  resource_size_t start = pci_resource_start(dev, bar);
  size_t size = (size_t)(__VERIFIER_nondet_ulong() & ~0xFFFUL);
  if (size == 0)
    size = 4096;
  return start + size - 1;
}

u32 pci_resource_flags(struct pci_dev *dev, int bar)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(bar >= 0 && bar < PCI_NUM_RESOURCES_WITH_ROM);
  return __VERIFIER_nondet_uint();
}

void *pci_iomap(struct pci_dev *dev, int bar, unsigned long max)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(bar >= 0 && bar < PCI_NUM_RESOURCES_WITH_ROM);

  resource_size_t start = pci_resource_start(dev, bar);
  if (start == 0)
    return NULL;

  if (max > ESBMC_MMIO_SPACE_SIZE)
    return NULL;

  /* Clamp so the whole window fits.  Written as a subtraction rather than
   * `offset + max > SIZE` because that sum overflows for a large offset and
   * wraps back under the limit, yielding an out-of-bounds mapping. */
  size_t offset = (size_t)(start % ESBMC_MMIO_SPACE_SIZE);
  if (offset > ESBMC_MMIO_SPACE_SIZE - max)
    offset = ESBMC_MMIO_SPACE_SIZE - max;

  return esbmc_mmio_space + offset;
}

void pci_iounmap(struct pci_dev *dev, void *addr)
{
  (void)dev;
  (void)addr;
}

int pci_enable_msi(struct pci_dev *dev)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = ENOSPC;
    return -1;
  }
  dev->irq = __VERIFIER_nondet_int() & 0xFFFF;
  return 0;
}

void pci_disable_msi(struct pci_dev *dev)
{
  (void)dev;
}

int pci_alloc_irq_vectors(
  struct pci_dev *dev,
  unsigned int min_vecs,
  unsigned int max_vecs,
  unsigned int flags)
{
__ESBMC_HIDE:;
  (void)flags;
  assert(dev != NULL);
  assert(min_vecs > 0);
  assert(max_vecs >= min_vecs);
  int result = __VERIFIER_nondet_int();
  if (result < 0)
    return -1;
  unsigned int vecs = __VERIFIER_nondet_int();
  __ESBMC_assume(vecs >= min_vecs && vecs <= max_vecs);
  return (int)vecs;
}

void pci_free_irq_vectors(struct pci_dev *dev)
{
  (void)dev;
}

int pci_read_config_byte(struct pci_dev *dev, int where, u8 *val)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(val != NULL);
  *val = __VERIFIER_nondet_uchar();
  return __VERIFIER_nondet_int();
}

int pci_read_config_word(struct pci_dev *dev, int where, u16 *val)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(val != NULL);
  *val = __VERIFIER_nondet_ushort();
  return __VERIFIER_nondet_int();
}

int pci_read_config_dword(struct pci_dev *dev, int where, u32 *val)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(val != NULL);
  *val = __VERIFIER_nondet_uint();
  return __VERIFIER_nondet_int();
}

int pci_write_config_byte(struct pci_dev *dev, int where, u8 val)
{
  (void)dev;
  (void)where;
  (void)val;
  return __VERIFIER_nondet_int();
}

int pci_write_config_word(struct pci_dev *dev, int where, u16 val)
{
  (void)dev;
  (void)where;
  (void)val;
  return __VERIFIER_nondet_int();
}

int pci_write_config_dword(struct pci_dev *dev, int where, u32 val)
{
  (void)dev;
  (void)where;
  (void)val;
  return __VERIFIER_nondet_int();
}

int pci_register_driver(struct pci_driver *drv)
{
__ESBMC_HIDE:;
  assert(drv != NULL);
  assert(drv->name != NULL);
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = ENODEV;
    return -1;
  }
  return 0;
}

void pci_unregister_driver(struct pci_driver *drv)
{
  (void)drv;
}

/* ============================================================
 *  PCIe AER (Advanced Error Reporting) models
 * ============================================================
 *
 * Models the AER capability structure and error handling path.
 * Reference: PCIe r4.0 §7.10, Linux drivers/pci/pcie/aer/
 * ============================================================ */

int pci_enable_aer(struct pci_dev *dev)
{
__ESBMC_HIDE:;
  assert(dev != NULL);

  dev->aer_enabled = 1;
  dev->aer_severity = AER_CORRECTABLE;
  dev->aer_corr_count = 0;
  dev->aer_non_fatal_count = 0;
  dev->aer_fatal_count = 0;
  return 0;
}

void pci_aer_clear(struct pci_dev *dev, int severity)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  if (!dev->aer_enabled)
    return;

  switch (severity)
  {
  case AER_CORRECTABLE:
    dev->aer_corr_count = 0;
    dev->aer_severity = AER_CORRECTABLE;
    break;
  case AER_NON_FATAL:
    dev->aer_non_fatal_count = 0;
    dev->aer_severity = AER_NON_FATAL;
    break;
  case AER_FATAL:
    dev->aer_fatal_count = 0;
    dev->aer_severity = AER_FATAL;
    break;
  default:
    dev->aer_corr_count = 0;
    dev->aer_non_fatal_count = 0;
    dev->aer_fatal_count = 0;
    break;
  }
}

int pci_aer_get_first_error(struct pci_dev *dev, int *severity)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(severity != NULL);

  if (!dev->aer_enabled)
    return -ENODEV;

  /* Return non-deterministic severity from available error types */
  int result = __VERIFIER_nondet_int();
  __ESBMC_assume(result >= 0 && result <= 2);
  *severity = result;
  return 0;
}

int pci_aer_clear_first_error(struct pci_dev *dev)
{
__ESBMC_HIDE:;
  assert(dev != NULL);

  if (!dev->aer_enabled)
    return -ENODEV;

  /* Clear and return the current severity */
  int severity = dev->aer_severity;
  dev->aer_corr_count = 0;
  dev->aer_non_fatal_count = 0;
  dev->aer_fatal_count = 0;
  dev->aer_severity = AER_CORRECTABLE;
  return severity;
}

/* ============================================================
 *  CXL device models
 * ============================================================ */

struct cxl_host_bridge *cxl_enumerate_ports(void)
{
__ESBMC_HIDE:;
  /* Non-deterministic: may or may not find a host bridge */
  if (__VERIFIER_nondet_int() == 0)
    return NULL;

  struct cxl_host_bridge *bridge = (struct cxl_host_bridge *)kmalloc(
    sizeof(struct cxl_host_bridge), GFP_KERNEL);
  if (bridge == NULL)
    return NULL;

  /* The array is always allocated at full size and only `num_devices` of it
   * is reported as populated.  Sizing the allocation by the symbolic count
   * instead would make every later `devices[i]` a bounds check against a
   * nondeterministically sized object, which does not solve: a four-device
   * walk did not finish in 280s that way, and finishes in under a second
   * this way.  A bridge with no devices still gets no array, because
   * kmalloc(0) trips kmalloc()'s own precondition. */
  unsigned int n = __VERIFIER_nondet_uint();
  __ESBMC_assume(n <= CXL_MAX_DOWNSTREAM_PORTS);
  bridge->num_devices = n;
  bridge->devices = NULL;
  if (n == 0)
    return bridge;

  bridge->devices = (struct cxl_dev *)kmalloc(
    CXL_MAX_DOWNSTREAM_PORTS * sizeof(struct cxl_dev), GFP_KERNEL);
  if (bridge->devices == NULL)
  {
    bridge->num_devices = 0;
    return bridge;
  }

  for (unsigned int i = 0; i < CXL_MAX_DOWNSTREAM_PORTS; i++)
  {
    unsigned int t = __VERIFIER_nondet_uint();
    __ESBMC_assume(t >= CXL_TYPE_FPMEM && t <= CXL_TYPE_RAM);
    bridge->devices[i].dev_type = t;
    bridge->devices[i].regs = esbmc_mmio_space;
    bridge->devices[i].port = NULL;
  }

  return bridge;
}

void cxl_free_ports(struct cxl_host_bridge *bridge)
{
  if (bridge == NULL)
    return;
  kfree(bridge->devices);
  kfree(bridge);
}

struct cxl_dev *
cxl_find_device(struct cxl_host_bridge *bridge, u16 vendor, u16 device)
{
__ESBMC_HIDE:;
  assert(bridge != NULL);
  (void)vendor;
  (void)device;
  if (bridge->num_devices == 0)
    return NULL;
  /* Constrain rather than take a modulus: __VERIFIER_nondet_int() may be
   * negative, and C's % keeps the sign of the dividend. */
  unsigned int idx = __VERIFIER_nondet_uint();
  __ESBMC_assume(idx < bridge->num_devices);
  return &bridge->devices[idx];
}

int cxl_device_init(struct cxl_dev *cxld)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);

  /* Read DEV_CTRL register — non-deterministic initial state */
  u64 ctrl = cxl_read_dev_ctrl(cxld);

  /* Assume device is in a valid initial state */
  __ESBMC_assume((ctrl & (CXL_DCR_CLEAR_INIT | CXL_DCR_ENABLE)) == 0);

  /* Set INIT bit */
  cxl_write_dev_ctrl(cxld, ctrl | CXL_DCR_CLEAR_INIT);

  /* Simulate device completing init — non-deterministic success/failure */
  int init_done = __VERIFIER_nondet_int();
  if (init_done != 0)
  {
    /* Clear INIT, set ENABLE */
    cxl_write_dev_ctrl(cxld, (ctrl & ~CXL_DCR_CLEAR_INIT) | CXL_DCR_ENABLE);
    return 0;
  }
  return -EIO;
}

void cxl_device_exit(struct cxl_dev *cxld)
{
  if (cxld == NULL)
    return;
  /* Disable the device */
  cxl_write_dev_ctrl(cxld, 0);
}

/* ============================================================
 *  CXL register access helpers
 * ============================================================ */

u64 cxl_read_dev_ctrl(struct cxl_dev *cxld)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  assert(cxld->regs != NULL);
  return (u64)readl(cxld->regs);
}

void cxl_write_dev_ctrl(struct cxl_dev *cxld, u64 val)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  assert(cxld->regs != NULL);
  writeq((uint64_t)val, cxld->regs);
}

u64 cxl_read_dev_stat(struct cxl_dev *cxld)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  assert(cxld->regs != NULL);
  return (u64)readl((char *)cxld->regs + 8);
}

/* ============================================================
 *  CXL Mailbox command model
 * ============================================================ */

int cxl_mailbox_send_cmd(struct cxl_dev *cxld, struct cxl_mailbox_cmd *cmd)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  assert(cmd != NULL);

  /* Validate opcode */
  __ESBMC_assume(cmd->opcode >= 0x0001 && cmd->opcode <= 0x4004);

  /* Write command to MMIO mailbox register */
  writel(cmd->opcode, (char *)cxld->regs + 0x100);

  /* Non-deterministic: command succeeds or fails */
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    cmd->status = __VERIFIER_nondet_uint();
    return -EIO;
  }

  /* Success: fill output payload with non-deterministic data */
  cmd->status = 0;
  if (cmd->payload_out != NULL && cmd->payload_out_size > 0)
  {
    memset(cmd->payload_out, 0, cmd->payload_out_size);
    /* First 4 bytes are non-deterministic */
    ((uint32_t *)cmd->payload_out)[0] = __VERIFIER_nondet_uint();
  }

  return 0;
}

/* ============================================================
 *  CXL Security models
 * ============================================================ */

enum cxl_security_state cxl_get_security_state(struct cxl_dev *cxld)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  int state = __VERIFIER_nondet_int();
  __ESBMC_assume(state >= CXL_SEC_NONE && state <= CXL_SEC_PASSPHRASE_SET);
  return (enum cxl_security_state)state;
}

int cxl_set_security(struct cxl_dev *cxld, enum cxl_security_state state)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  __ESBMC_assume(state >= CXL_SEC_NONE && state <= CXL_SEC_PASSPHRASE_SET);

  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = EPERM;
    return -1;
  }
  return 0;
}

/* ============================================================
 *  CXL error injection models
 * ============================================================
 *
 * Models CXL error injection via the mailbox command interface.
 * Reference: CXL 2.0 §8.2.9.4 (Error Record Serialization)
 * ============================================================ */

static int esbmc_corr_count = 0;
static int esbmc_non_fatal_count = 0;
static int esbmc_fatal_count = 0;

int cxl_err_inject(struct cxl_dev *cxld, enum cxl_error_type type)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  __ESBMC_assume(type >= CXL_ERR_CORRECTABLE && type <= CXL_ERR_FATAL);

  /* Non-deterministic: injection hardware may not be present */
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = ENODEV;
    return -1;
  }

  /* Record the error */
  switch (type)
  {
  case CXL_ERR_CORRECTABLE:
    esbmc_corr_count++;
    break;
  case CXL_ERR_NON_FATAL:
    esbmc_non_fatal_count++;
    break;
  case CXL_ERR_FATAL:
    esbmc_fatal_count++;
    break;
  }
  return 0;
}

int cxl_err_get_count(
  struct cxl_dev *cxld,
  int *correctable,
  int *non_fatal,
  int *fatal)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  assert(correctable != NULL);
  assert(non_fatal != NULL);
  assert(fatal != NULL);

  *correctable = esbmc_corr_count;
  *non_fatal = esbmc_non_fatal_count;
  *fatal = esbmc_fatal_count;
  return 0;
}

/* ============================================================
 *  CXL HDM decoder setup
 * ============================================================
 *
 * CXL 2.0 §8.2.2.12:
 *   - Maximum 8 HDM decoders per device (CXL_HDM_DECODER_MAX)
 *   - Region base address must be aligned to 4KB minimum
 *     (CXL_HDM_ALIGNMENT)
 * ============================================================ */

int cxl_setup_hdm_decoders(
  struct cxl_dev *cxld,
  const struct cxl_region *region)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  assert(region != NULL);
  assert(region->size > 0);
  assert(region->granularity > 0);

  /* Validate 4KB alignment on region base address (CXL 2.0 §8.2.2.12.1) */
  if ((region->start % CXL_HDM_ALIGNMENT) != 0)
  {
    errno = EINVAL;
    return -1;
  }

  /* Enforce 8-decoder limit (CXL 2.0 §8.2.2.12) */
  /* Non-deterministic: the device may have 0-7 decoders already */
  int existing = __VERIFIER_nondet_int();
  __ESBMC_assume(existing >= 0 && existing < CXL_HDM_DECODER_MAX);

  if (existing >= (CXL_HDM_DECODER_MAX - 1))
  {
    errno = ENOSPC;
    return -1;
  }

  /* Non-deterministic: decoder setup may still fail for other reasons */
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = ENOSPC;
    return -1;
  }
  return 0;
}

/* ============================================================
 *  CXL driver registration
 * ============================================================ */

int cxl_driver_register(struct cxl_driver *drv)
{
__ESBMC_HIDE:;
  assert(drv != NULL);
  assert(drv->name != NULL);
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = EINVAL;
    return -1;
  }
  return 0;
}

void cxl_driver_unregister(struct cxl_driver *drv)
{
  (void)drv;
}

/* ============================================================
 *  CXL Memory device models
 * ============================================================ */

struct cxl_mem *cxl_mem_attach(struct cxl_dev *cxld)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  struct cxl_mem *cxlmem =
    (struct cxl_mem *)kmalloc(sizeof(struct cxl_mem), GFP_KERNEL);
  if (cxlmem == NULL)
    return NULL;

  cxlmem->cxld = cxld;
  cxlmem->capabilities_off = __VERIFIER_nondet_uint();
  cxlmem->mbox_off = 0x100;
  cxlmem->dw8_size = 4096;
  cxlmem->payload_max = 4096;

  return cxlmem;
}

void cxl_mem_detach(struct cxl_mem *cxlmem)
{
  if (cxlmem == NULL)
    return;
  kfree(cxlmem);
}

void cxl_mem_flush(struct cxl_mem *cxlmem)
{
__ESBMC_HIDE:;
  assert(cxlmem != NULL);
  /* Flush is a no-op in the model (ordering handled by barriers) */
}

int cxl_mem_enable(struct cxl_mem *cxlmem)
{
__ESBMC_HIDE:;
  assert(cxlmem != NULL);
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = EIO;
    return -1;
  }
  return 0;
}

void cxl_mem_disable(struct cxl_mem *cxlmem)
{
  if (cxlmem == NULL)
    return;
  writel(0, (char *)cxlmem->cxld->regs + 0x400);
}

int cxl_mem_get_regions(
  struct cxl_mem *cxlmem,
  struct cxl_memregion_info *regions,
  unsigned int max_regions)
{
__ESBMC_HIDE:;
  assert(cxlmem != NULL);
  assert(regions != NULL);
  assert(max_regions > 0);

  /* Non-deterministic number of regions (1 to max_regions) */
  /* Constrain rather than take a modulus of a signed nondet: C's % keeps the
   * sign of the dividend, so `nondet_int() % max + 1` reaches 0 and below. */
  unsigned int n = __VERIFIER_nondet_uint();
  __ESBMC_assume(n >= 1 && n <= max_regions);

  for (unsigned int i = 0; i < n; i++)
  {
    regions[i].index = i;
    regions[i].base = (u64)(__VERIFIER_nondet_ulong() & ~0xFFFUL);
    regions[i].size = (u64)(__VERIFIER_nondet_ulong() & ~0xFFFUL);
    if (regions[i].size == 0)
      regions[i].size = 4096;
    regions[i].phys_handle = 0;
    regions[i].mapping = __VERIFIER_nondet_uchar() % 8;
  }

  return (int)n;
}

int cxl_mem_set_pmem_capacity(struct cxl_mem *cxlmem, u64 size)
{
__ESBMC_HIDE:;
  assert(cxlmem != NULL);
  assert(size > 0);
  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = EINVAL;
    return -1;
  }
  return 0;
}

int cxl_mem_get_partition_state(
  struct cxl_mem *cxlmem,
  u32 *split_data_size,
  u32 *split_pmem_size)
{
__ESBMC_HIDE:;
  assert(cxlmem != NULL);
  assert(split_data_size != NULL);
  assert(split_pmem_size != NULL);

  *split_data_size = __VERIFIER_nondet_uint();
  *split_pmem_size = __VERIFIER_nondet_uint();
  return 0;
}

int cxl_mem_set_partition_state(
  struct cxl_mem *cxlmem,
  u32 split_data_size,
  u32 split_pmem_size)
{
__ESBMC_HIDE:;
  assert(cxlmem != NULL);

  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = EINVAL;
    return -1;
  }
  return 0;
}

/* ============================================================
 *  DMA models
 * ============================================================ */

void *dma_alloc_coherent(
  struct device *dev,
  size_t size,
  dma_addr_t *dma_handle,
  unsigned int flag)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(size > 0);
  assert(size <= ESBMC_DMA_SPACE_SIZE);

  /* Validate GFP flags */
  check_gfp_flags(flag);

  /* Find a free region in DMA space */
  int offset = -1;
  for (int i = 0; i <= (int)(ESBMC_DMA_SPACE_SIZE - size); i++)
  {
    if (!esbmc_dma_used[i])
    {
      offset = i;
      break;
    }
  }

  if (offset < 0)
    return NULL; /* No free DMA space */

  /* Mark region as used */
  for (size_t i = 0; i < size; i++)
    esbmc_dma_used[offset + i] = 1;

  *dma_handle = (dma_addr_t)offset;

  /* Return CPU-visible address in DMA space */
  return esbmc_dma_space + offset;
}

void dma_free_coherent(
  struct device *dev,
  size_t size,
  void *cpu_addr,
  dma_addr_t dma_handle)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(size > 0);
  assert(cpu_addr != NULL);
  assert((char *)cpu_addr >= esbmc_dma_space);
  assert((char *)cpu_addr < esbmc_dma_space + ESBMC_DMA_SPACE_SIZE);

  int offset = (int)((char *)cpu_addr - esbmc_dma_space);
  for (size_t i = 0; i < size; i++)
    esbmc_dma_used[offset + i] = 0;
}

dma_addr_t dma_map_single(
  struct device *dev,
  void *cpu_addr,
  size_t size,
  enum dma_data_direction dir)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(cpu_addr != NULL);
  assert(size > 0);
  assert(dir >= DMA_BIDIRECTIONAL && dir <= DMA_NONE);

  /* Return a non-deterministic DMA address */
  return (dma_addr_t)__VERIFIER_nondet_ulong();
}

void dma_unmap_single(
  struct device *dev,
  dma_addr_t dma_handle,
  size_t size,
  enum dma_data_direction dir)
{
  (void)dev;
  (void)dma_handle;
  (void)size;
  (void)dir;
}

void dma_sync_single_for_cpu(
  struct device *dev,
  dma_addr_t dma_handle,
  size_t size,
  enum dma_data_direction dir)
{
  (void)dev;
  (void)dma_handle;
  (void)size;
  (void)dir;
}

void dma_sync_single_for_device(
  struct device *dev,
  dma_addr_t dma_handle,
  size_t size,
  enum dma_data_direction dir)
{
  (void)dev;
  (void)dma_handle;
  (void)size;
  (void)dir;
}

int dma_set_mask(struct device *dev, uint64_t mask)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(mask > 0);
  return __VERIFIER_nondet_int();
}

int dma_set_coherent_mask(struct device *dev, uint64_t mask)
{
__ESBMC_HIDE:;
  assert(dev != NULL);
  assert(mask > 0);
  return __VERIFIER_nondet_int();
}

/* ============================================================
 *  IRQ models
 * ============================================================ */

int request_irq(
  unsigned int irq,
  irq_handler_t handler,
  unsigned long flags,
  const char *name,
  void *dev_id)
{
__ESBMC_HIDE:;
  assert(handler != NULL);
  assert(name != NULL);
  assert(dev_id != NULL);

  if (esbmc_irq_count >= ESBMC_IRQ_HANDLERS_MAX)
  {
    errno = ENOSPC;
    return -1;
  }

  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = EBUSY;
    return -1;
  }

  esbmc_irq_table[esbmc_irq_count].irq = irq;
  esbmc_irq_table[esbmc_irq_count].handler = handler;
  esbmc_irq_table[esbmc_irq_count].dev_id = dev_id;
  esbmc_irq_count++;

  return 0;
}

void free_irq(unsigned int irq, void *dev_id)
{
__ESBMC_HIDE:;
  assert(dev_id != NULL);

  for (int i = 0; i < esbmc_irq_count; i++)
  {
    if (esbmc_irq_table[i].irq == irq && esbmc_irq_table[i].dev_id == dev_id)
    {
      /* Remove entry by shifting */
      esbmc_irq_count--;
      for (int j = i; j < esbmc_irq_count; j++)
      {
        esbmc_irq_table[j] = esbmc_irq_table[j + 1];
      }
      return;
    }
  }
}

void disable_irq(unsigned int irq)
{
  (void)irq;
}

void enable_irq(unsigned int irq)
{
  (void)irq;
}

void disable_irq_nosync(unsigned int irq)
{
  (void)irq;
}

void synchronize_irq(unsigned int irq)
{
  (void)irq;
}

void mask_irq(unsigned int irq)
{
  (void)irq;
}

void unmask_irq(unsigned int irq)
{
  (void)irq;
}

void esbmc_simulate_irq(unsigned int irq, void *dev_id)
{
__ESBMC_HIDE:;
  /* Find and invoke the registered handler */
  for (int i = 0; i < esbmc_irq_count; i++)
  {
    if (esbmc_irq_table[i].irq == irq && esbmc_irq_table[i].dev_id == dev_id)
    {
      esbmc_irq_table[i].handler(irq, dev_id);
      return;
    }
  }
}

/* ============================================================
 *  CXL memdev (/dev/cxl/memN) models
 * ============================================================
 *
 * drivers/cxl/core/memdev.c draws a minor number per memory expander
 * from an IDA.  That allocation is fallible, but the failure only
 * shows up once the id space is exhausted, so drivers routinely skip
 * the check.  The model keeps exhaustion reachable on every call so
 * the missing check is caught.
 * ============================================================ */

int cxl_memdev_id_alloc(void)
{
__ESBMC_HIDE:;
  int id = __VERIFIER_nondet_int();
  if (id < 0 || id >= CXL_MEM_MAX_DEVS)
    return -ENOSPC;
  return id;
}

struct cxl_memdev *cxl_memdev_create(struct cxl_dev *cxld)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);

  int id = cxl_memdev_id_alloc();
  if (id < 0)
    return NULL;

  struct cxl_memdev *cxlmd =
    (struct cxl_memdev *)kmalloc(sizeof(struct cxl_memdev), GFP_KERNEL);
  if (cxlmd == NULL)
    return NULL;

  cxlmd->cxld = cxld;
  cxlmd->id = id;
  cxlmd->live = 1;

  /* IDENTIFY returns an opaque revision string, so the buffer keeps the
     non-deterministic contents kmalloc left behind.  The one guarantee a
     driver may rely on is termination. */
  cxlmd->fw_rev[CXL_MEMDEV_FW_REV_LEN - 1] = '\0';

  return cxlmd;
}

void cxl_memdev_destroy(struct cxl_memdev *cxlmd)
{
  if (cxlmd == NULL)
    return;
  cxlmd->live = 0;
  kfree(cxlmd);
}

/* ============================================================
 *  CXL region interleave models
 * ============================================================
 *
 * CXL 3.0 §8.2.4.20.1 encodes granularity and ways as exponents, so
 * both must be powers of two inside their ranges;
 * drivers/cxl/core/region.c rejects anything else in
 * granularity_to_eig() and ways_to_eiw().
 * ============================================================ */

static int esbmc_is_power_of_two(unsigned int v)
{
  return v != 0 && (v & (v - 1)) == 0;
}

int cxl_region_config(
  struct cxl_region *region,
  unsigned int ways,
  unsigned int granularity)
{
__ESBMC_HIDE:;
  assert(region != NULL);

  if (!esbmc_is_power_of_two(ways) || ways > CXL_DECODER_MAX_WAYS)
  {
    errno = EINVAL;
    return -1;
  }

  if (
    !esbmc_is_power_of_two(granularity) ||
    granularity < CXL_DECODER_MIN_GRANULARITY ||
    granularity > CXL_DECODER_MAX_GRANULARITY)
  {
    errno = EINVAL;
    return -1;
  }

  /* Every interleave position must own a whole number of stripes,
     otherwise one position holds a partial stripe. */
  resource_size_t stripe = (resource_size_t)ways * granularity;
  if (region->size == 0 || (region->size % stripe) != 0)
  {
    errno = EINVAL;
    return -1;
  }

  region->ways = ways;
  region->granularity = granularity;
  return 0;
}

int cxl_region_overlaps(const struct cxl_region *a, const struct cxl_region *b)
{
__ESBMC_HIDE:;
  assert(a != NULL);
  assert(b != NULL);

  /* A host physical address range cannot wrap the address space. The
     half-open test below adds start to size, so without this precondition
     a wrapping range would silently report "no overlap". */
  assert(a->start <= (resource_size_t)-1 - a->size);
  assert(b->start <= (resource_size_t)-1 - b->size);

  return a->start < b->start + b->size && b->start < a->start + a->size;
}

/* ============================================================
 *  CXL mailbox IOCTL models
 * ============================================================
 *
 * drivers/cxl/core/mbox.c validates a user command in three steps: the
 * opcode must appear in cxl_mem_commands[], the payload must fit, and the
 * device must have that command enabled. All three rejections stay
 * reachable here.
 * ============================================================ */

static const u16 esbmc_cxl_mbox_opcodes[] = {
  CXL_MBOX_OP_GET_SUPPORTED_LOGS,
  CXL_MBOX_OP_GET_CAPABILITIES,
  CXL_MBOX_OP_GET_STATUS,
  CXL_MBOX_OP_SET_SECURITY,
  CXL_MBOX_OP_SET_LOCK,
  CXL_MBOX_OP_SET_PMEM_CAP,
  CXL_MBOX_OP_GET_PARTITION_STATE,
  CXL_MBOX_OP_SET_PARTITION_STATE,
};

#define ESBMC_CXL_MBOX_OPCODE_COUNT                                            \
  (sizeof(esbmc_cxl_mbox_opcodes) / sizeof(esbmc_cxl_mbox_opcodes[0]))

int cxl_mbox_cmd_index(u16 opcode)
{
__ESBMC_HIDE:;
  for (unsigned int i = 0; i < ESBMC_CXL_MBOX_OPCODE_COUNT; i++)
    if (esbmc_cxl_mbox_opcodes[i] == opcode)
      return (int)i;
  return -1;
}

int cxl_mailbox_ioctl(struct cxl_dev *cxld, u16 opcode, void *payload, u32 size)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);

  if (cxl_mbox_cmd_index(opcode) < 0)
    return -ENOTTY;

  if (size > CXL_MBOX_IOCTL_MAX_PAYLOAD)
    return -EINVAL;

  if (size > 0 && payload == NULL)
    return -EINVAL;

  /* Present in the table, but the device need not have it enabled. */
  if (!__VERIFIER_nondet_int())
    return -ENOTTY;

  return 0;
}

/* ============================================================
 *  CXL downstream port (dport) models
 * ============================================================
 *
 * A dport lives exactly as long as its parent port keeps it registered;
 * Linux drops it in devm cleanup. Removal frees the object here too, so a
 * pointer cached across a removal is a use-after-free rather than a
 * silently stale read.
 * ============================================================ */

static struct cxl_dport *esbmc_cxl_dports[CXL_PORT_MAX_DPORTS];

struct cxl_dport *cxl_dport_find(struct cxl_port *port, int id)
{
__ESBMC_HIDE:;
  assert(port != NULL);
  for (int i = 0; i < CXL_PORT_MAX_DPORTS; i++)
    if (
      esbmc_cxl_dports[i] != NULL && esbmc_cxl_dports[i]->port == port &&
      esbmc_cxl_dports[i]->id == id)
      return esbmc_cxl_dports[i];
  return NULL;
}

int cxl_dport_add(struct cxl_port *port, struct pci_dev *dport_dev, int id)
{
__ESBMC_HIDE:;
  assert(port != NULL);

  if (cxl_dport_find(port, id) != NULL)
    return -EBUSY;

  for (int i = 0; i < CXL_PORT_MAX_DPORTS; i++)
  {
    if (esbmc_cxl_dports[i] != NULL)
      continue;

    struct cxl_dport *dp =
      (struct cxl_dport *)kmalloc(sizeof(struct cxl_dport), GFP_KERNEL);
    if (dp == NULL)
      return -ENOMEM;

    dp->port = port;
    dp->dport_dev = dport_dev;
    dp->id = id;
    esbmc_cxl_dports[i] = dp;
    return 0;
  }
  return -ENOSPC;
}

void cxl_dport_remove(struct cxl_port *port, int id)
{
__ESBMC_HIDE:;
  assert(port != NULL);
  for (int i = 0; i < CXL_PORT_MAX_DPORTS; i++)
    if (
      esbmc_cxl_dports[i] != NULL && esbmc_cxl_dports[i]->port == port &&
      esbmc_cxl_dports[i]->id == id)
    {
      kfree(esbmc_cxl_dports[i]);
      esbmc_cxl_dports[i] = NULL;
      return;
    }
}

int cxl_dport_count(struct cxl_port *port)
{
__ESBMC_HIDE:;
  assert(port != NULL);
  int n = 0;
  for (int i = 0; i < CXL_PORT_MAX_DPORTS; i++)
    if (esbmc_cxl_dports[i] != NULL && esbmc_cxl_dports[i]->port == port)
      n++;
  return n;
}
