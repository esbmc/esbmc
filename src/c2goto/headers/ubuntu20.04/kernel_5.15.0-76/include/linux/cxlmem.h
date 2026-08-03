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

/* ============================================================
 *  CXL PMEM security — declared here, modelled in cxl_driver.c
 * ============================================================
 *
 * Mirrors drivers/cxl/security.c against the real device flags in
 * drivers/cxl/cxlmem.h.  Unlike the invented cxl_security_state enum in
 * cxl.h, these are the actual CXL 2.0 §8.2.9.8.6 Get Security State bits
 * and the actual nvdimm flags security.c derives from them.
 */

/* drivers/cxl/cxlmem.h: device-reported security state */
#define CXL_PMEM_SEC_STATE_USER_PASS_SET   0x01
#define CXL_PMEM_SEC_STATE_MASTER_PASS_SET 0x02
#define CXL_PMEM_SEC_STATE_LOCKED          0x04
#define CXL_PMEM_SEC_STATE_FROZEN          0x08
#define CXL_PMEM_SEC_STATE_USER_PLIMIT     0x10
#define CXL_PMEM_SEC_STATE_MASTER_PLIMIT   0x20

/* include/linux/libnvdimm.h: flags security.c reports upwards */
#define NVDIMM_SECURITY_DISABLED  0x01
#define NVDIMM_SECURITY_UNLOCKED  0x02
#define NVDIMM_SECURITY_LOCKED    0x04
#define NVDIMM_SECURITY_FROZEN    0x08

/* nvdimm_passphrase_type */
#define NVDIMM_USER   0
#define NVDIMM_MASTER 1

/* Maximum passphrase length (CXL 2.0 §8.2.9.8.6.2) */
#define CXL_PMEM_PASSPHRASE_LEN 32

struct cxl_pmem_security
{
  u32 state;                                /* CXL_PMEM_SEC_STATE_* */
  char user_pass[CXL_PMEM_PASSPHRASE_LEN];  /* valid iff USER_PASS_SET */
  char master_pass[CXL_PMEM_PASSPHRASE_LEN];/* valid iff MASTER_PASS_SET */
};

/*
 * Derives the nvdimm security flags from the device-reported state, exactly
 * as cxl_pmem_get_security_state() does.  ptype selects the USER or MASTER
 * view; they report independently.
 */
unsigned long cxl_pmem_security_flags(u32 sec_state, int ptype);

/*
 * Passphrase operations.  Each returns 0 or a negative errno, and each is
 * refused when the device state forbids it -- a frozen device accepts no
 * passphrase change, and an unlock needs the passphrase to match.
 */
int cxl_pmem_set_passphrase(struct cxl_pmem_security *sec, int ptype,
                            const char *old_pass, const char *new_pass);
int cxl_pmem_unlock(struct cxl_pmem_security *sec, const char *pass);
int cxl_pmem_freeze(struct cxl_pmem_security *sec);

#endif /* _LINUX_CXLMEM_H */
