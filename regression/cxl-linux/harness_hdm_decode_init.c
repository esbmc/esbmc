#include <linux/compiler-version.h>
#include <linux/kconfig.h>
#include <linux/compiler_types.h>

#include "drivers/cxl/core/pci.c"

int __VERIFIER_nondet_int(void);

/* cxl_hdm_decode_init() walks info->dvsec_range[i] for i < info->ranges but
 * never re-validates info->ranges against the array's 2 entries. Its only
 * caller (core/hdm.c:1273) passes the info that cxl_dvsec_rr_decode()
 * populated at hdm.c:1262, which caps ranges at 2 -- an unstated precondition.
 * RANGES_BOUNDED models that caller contract. */

/* to_cxl_port() lives in core/port.c. Model its success path: it returns
 * container_of(dev, struct cxl_port, dev) once the device type checks out.
 * (On the failure path it returns NULL and the caller here would dereference
 * it -- out of scope for this harness, which fixes a well-typed topology.) */
struct cxl_port *to_cxl_port(const struct device *dev)
{
  return container_of(dev, struct cxl_port, dev);
}

/* The match callback resolves the child device to a decoder and reads its
 * hpa_range. Both accessors live in core/port.c; model a well-formed decoder
 * so the callback reaches range_contains() and actually reads the range it
 * was handed. */
static struct cxl_decoder harness_decoder;

struct cxl_decoder *to_cxl_decoder(struct device *dev)
{
  return &harness_decoder;
}

/* Model the real iterator: it invokes match() on each child, and the CXL
 * match callback (dvsec_range_allowed) reads the range it is handed. Stubbing
 * this out entirely would leave &info->dvsec_range[i] as mere address
 * arithmetic and the out-of-bounds read would never be exercised. */
struct device *device_find_child(
  struct device *parent,
  const void *data,
  device_match_t match)
{
  static struct device child;
  if (match(&child, data))
    return &child;
  return NULL;
}

int main(void)
{
  static struct pci_dev pdev;
  static struct cxl_dev_state cxlds;
  static struct cxl_hdm cxlhdm;
  static struct cxl_port port;
  static struct cxl_port root_port;
  static u32 hdm_regs[64];
  struct cxl_endpoint_dvsec_info info = {};

  cxlds.dev = &pdev.dev;
  cxlhdm.regs.hdm_decoder = (void __iomem *)hdm_regs;
  cxlhdm.port = &port;

  /* Parent of the endpoint port is the root port; zero-initialised
   * uport_dev == dev.parent makes is_cxl_root() true. */
  port.dev.parent = &root_port.dev;

  /* Without CXL_DECODER_F_RAM the match callback returns before
   * range_contains(), so the range would never actually be read and the
   * harness would prove nothing. */
  harness_decoder.flags = CXL_DECODER_F_RAM;

  info.mem_enabled = true;
  info.ranges = __VERIFIER_nondet_int();
  __ESBMC_assume(info.ranges >= 0);
#ifdef RANGES_BOUNDED
  __ESBMC_assume(info.ranges <= 2);
#else
  __ESBMC_assume(info.ranges <= 4);
#endif

  int rc = cxl_hdm_decode_init(&cxlds, &cxlhdm, &info);
  (void)rc;
  return 0;
}
