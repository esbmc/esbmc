/* SPDX-License-Identifier: GPL-2.0
 *
 * linux/cxl.h — CXL device driver core header (ESBMC operational model stubs).
 *
 * This header provides declarations for the CXL core driver API surface.
 * Function bodies are modeled in src/c2goto/library/cxl_driver.c.
 */
#ifndef _LINUX_CXL_H
#define _LINUX_CXL_H

#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/asm/io.h>

/* CXL device types per the CXL 2.0/3.0 spec */
#define CXL_TYPE_FPMEM  1
#define CXL_TYPE_PMEM   2
#define CXL_TYPE_RAM    3

/* CXL device flags */
#define CXL_DEV_FLAG_IS_FPMEM  (1 << 0)
#define CXL_DEV_FLAG_IS_PMEM   (1 << 1)
#define CXL_DEV_FLAG_IS_RAM    (1 << 2)

/* CXL region geometry */
struct cxl_region {
  resource_size_t start;
  resource_size_t size;
  unsigned int granularity;
};

/* CXL port (root port / switch port) */
struct cxl_port {
  struct pci_dev *pdev;
  u32 port_type;    /* 0 = root, 1 = downstream, 2 = upstream */
  u32 rtype;        /* register type */
};

/* CXL device — the central abstraction */
struct cxl_dev {
  struct pci_dev *pdev;
  u32 dev_type;     /* CXL_TYPE_FPMEM / PMEM / RAM */
  u32 flags;
  void *regs;   /* MMIO base for device registers */
  struct cxl_port *port;
  struct cxl_region region;
};

/* CXL host bridge — top-level container */
struct cxl_host_bridge {
  struct pci_dev *pdev;
  struct cxl_dev **devices;
  unsigned int num_devices;
};

/* CXL driver — device driver registration */
struct cxl_driver {
  const char *name;
  int (*probe)(struct cxl_dev *cxld, const struct cxl_device_id *id);
  void (*remove)(struct cxl_dev *cxld);
  const struct cxl_device_id *ids;
  unsigned int nids;
};

/* Device ID table (PCI-style) */
struct cxl_device_id {
  u16 vendor;
  u16 device;
  u32 subclass;
  unsigned long driver_data;
};

/* CXL register offsets (CXL 2.0 spec) */
#define CXL_REGMAP_DEV_CTRL   0x0000
#define CXL_REGMAP_DEV_STAT   0x0008
#define CXL_REGMAP_MAILBOX   0x0100
#define CXL_REGMAP_ISA       0x0200
#define CXL_REGMAP_HDM_DEC   0x0300

/* CXL Device Control register bits */
#define CXL_DCR_CLEAR_INIT    (1 << 0)
#define CXL_DCR_ENABLE        (1 << 1)
#define CXL_DCR_RESET         (1 << 2)

/* CXL Device Status register bits */
#define CXL_DSR_INIT_DONE     (1 << 0)
#define CXL_DSR_ENABLED       (1 << 1)
#define CXL_DSR_HAS_ERROR     (1 << 2)

/* CXL Mailbox command opcodes (CXL 2.0 §8.1.3) */
#define CXL_MBOX_OP_GET_SUPPORTED_LOGS  0x0001
#define CXL_MBOX_OP_GET_CAPABILITIES    0x0002
#define CXL_MBOX_OP_GET_STATUS          0x0003
#define CXL_MBOX_OP_SET_SECURITY        0x0005
#define CXL_MBOX_OP_SET_LOCK            0x0006
#define CXL_MBOX_OP_SET_PMEM_CAP        0x0007
#define CXL_MBOX_OP_GET_PARTITION_STATE 0x4001
#define CXL_MBOX_OP_SET_PARTITION_STATE 0x4002
#define CXL_MBOX_OP_GET_HDM_DECoders    0x4003
#define CXL_MBOX_OP_SET_HDM_DECoders    0x4004

/* CXL mailbox command payload */
struct cxl_mailbox_cmd {
  u16 opcode;
  u16 payload_in_size;
  u16 payload_out_size;
  void *payload_in;
  void *payload_out;
  u32 status;
};

/* CXL security state */
enum cxl_security_state {
  CXL_SEC_NONE = 0,
  CXL_SEC_UNLOCKED,
  CXL_SEC_LOCKED,
  CXL_SEC_DISABLED,
  CXL_SEC_PASSPHRASE_SET,
};

/* ============================================================
 *  Core CXL API — declared here, modelled in cxl_driver.c
 * ============================================================ */

/* Device enumeration */
struct cxl_host_bridge *cxl_enumerate_ports(void);
void cxl_free_ports(struct cxl_host_bridge *bridge);
struct cxl_dev *cxl_find_device(struct cxl_host_bridge *bridge, u16 vendor,
                                u16 device);

/* Device lifecycle */
int cxl_device_init(struct cxl_dev *cxld);
void cxl_device_exit(struct cxl_dev *cxld);

/* Mailbox command submission */
int cxl_mailbox_send_cmd(struct cxl_dev *cxld, struct cxl_mailbox_cmd *cmd);

/* Device control register access */
u64 cxl_read_dev_ctrl(struct cxl_dev *cxld);
void cxl_write_dev_ctrl(struct cxl_dev *cxld, u64 val);

/* Device status register access */
u64 cxl_read_dev_stat(struct cxl_dev *cxld);

/* Security operations */
enum cxl_security_state cxl_get_security_state(struct cxl_dev *cxld);
int cxl_set_security(struct cxl_dev *cxld, enum cxl_security_state state);

/* HDM decoder setup (host memory decode) */
int cxl_setup_hdm_decoders(struct cxl_dev *cxld,
                           const struct cxl_region *region);

/* Driver registration */
int cxl_driver_register(struct cxl_driver *drv);
void cxl_driver_unregister(struct cxl_driver *drv);

#endif /* _LINUX_CXL_H */
