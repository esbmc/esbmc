/* SPDX-License-Identifier: GPL-2.0
 *
 * linux/cxlmem.h — CXL memory device header (ESBMC operational model stubs).
 *
 * This header provides declarations for the CXL memory device driver API.
 * Function bodies are modeled in src/c2goto/library/cxl_driver.c.
 *
 * Synthetic, and not version-pinned: see the header comment in cxl.h for what
 * the kernel_5.15.0-76 path does and does not mean.
 */
#ifndef _LINUX_CXLMEM_H
#define _LINUX_CXLMEM_H

#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

/* CXL memory device register offsets */
#define CXLMEM_REGMAP_CAP   0x0000
#define CXLMEM_REGMAP_MBOX  0x0100
#define CXLMEM_REGMAP_ISO   0x0200
#define CXLMEM_REGMAP_DCR   0x0400

/* CXL memory device control register bits */
#define CXLMEM_DCR_FLUSH     (1 << 0)
#define CXLMEM_DCR_INIT      (1 << 1)
#define CXLMEM_DCR_ENABLE    (1 << 2)

/* CXL memory region type */
#define CXL_MEM_REGION_FPMEM 0  /* Fine-grained persistent memory */
#define CXL_MEM_REGION_PMEM  1  /* Coarse-grained persistent memory */

/* CXL memory device */
struct cxl_mem {
  struct cxl_dev *cxld;
  u32 capabilities_off;
  u32 mbox_off;
  u32 iso_off;
  u64 dw8_size;    /* 64-bit data window size */
  u64 payload_max; /* max mailbox payload */
};

/* CXL memory region info (from GET_CAPABILITIES log) */
struct cxl_memregion_info {
  u32 index;
  u64 base;
  u64 size;
  u64 phys_handle;
  u8 mapping;    /* which decoder maps this region */
};

/* CXL memory device driver */
struct cxl_mem_driver {
  const char *name;
  int (*probe)(struct cxl_mem *cxlmem);
  void (*remove)(struct cxl_mem *cxlmem);
  const struct cxl_device_id *ids;
  unsigned int nids;
};

/* ============================================================
 *  CXL Memory API — declared here, modelled in cxl_driver.c
 * ============================================================ */

/* Memory device initialization */
struct cxl_mem *cxl_mem_attach(struct cxl_dev *cxld);
void cxl_mem_detach(struct cxl_mem *cxlmem);

/* Memory device control */
void cxl_mem_flush(struct cxl_mem *cxlmem);
int cxl_mem_enable(struct cxl_mem *cxlmem);
void cxl_mem_disable(struct cxl_mem *cxlmem);

/* Query memory regions */
int cxl_mem_get_regions(struct cxl_mem *cxlmem,
                        struct cxl_memregion_info *regions,
                        unsigned int max_regions);

/* Set persistent memory capacity */
int cxl_mem_set_pmem_capacity(struct cxl_mem *cxlmem, u64 size);

/* Get partition state */
int cxl_mem_get_partition_state(struct cxl_mem *cxlmem,
                                u32 *split_data_size,
                                u32 *split_pmem_size);

/* Set partition state */
int cxl_mem_set_partition_state(struct cxl_mem *cxlmem,
                                u32 split_data_size,
                                u32 split_pmem_size);

/* ============================================================
 *  CXL memdev (/dev/cxl/memN) — declared here, modelled in
 *  cxl_driver.c
 * ============================================================
 *
 * Models the character device drivers/cxl/core/memdev.c creates for
 * each memory expander.  Real Linux spells the constructors
 * cxl_memdev_alloc() / devm_cxl_add_memdev() and draws the minor
 * number from ida_alloc_range(&cxl_memdev_ida, 0, CXL_MEM_MAX_DEVS,
 * GFP_KERNEL) — an allocation that can fail.
 */

/* drivers/cxl/cxlmem.h: CXL_MEM_MAX_DEVS */
#define CXL_MEM_MAX_DEVS 65536

/* FW revision string from IDENTIFY output (CXL 2.0 §8.2.9.5.1) */
#define CXL_MEMDEV_FW_REV_LEN 16

struct cxl_memdev
{
  struct cxl_dev *cxld;
  int id; /* N in /dev/cxl/memN; negative when unallocated */
  char fw_rev[CXL_MEMDEV_FW_REV_LEN]; /* always NUL-terminated */
  int live;                           /* non-zero between create and destroy */
};

/*
 * Returns a free minor number in [0, CXL_MEM_MAX_DEVS), or -ENOSPC when
 * the id space is exhausted.  Callers must check the sign before using
 * the result as an index.
 */
int cxl_memdev_id_alloc(void);

struct cxl_memdev *cxl_memdev_create(struct cxl_dev *cxld);
void cxl_memdev_destroy(struct cxl_memdev *cxlmd);

#endif /* _LINUX_CXLMEM_H */
