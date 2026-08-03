/* SPDX-License-Identifier: GPL-2.0
 *
 * linux/cxl.h — CXL device driver core header (ESBMC operational model stubs).
 *
 * This header provides declarations for the CXL core driver API surface.
 * Function bodies are modeled in src/c2goto/library/cxl_driver.c.
 *
 * The CXL API declared here is SYNTHETIC -- it is a CXL-like surface invented
 * for this work, not a copy of any kernel's. Linux has no include/linux/cxl.h
 * at all; the real declarations live in drivers/cxl/cxl.h, and the real names
 * differ (struct cxl_memdev / cxl_dev_state, cxl_internal_send_cmd(), ...).
 *
 * The kernel_5.15.0-76 path is therefore NOT a version pin for this file. It
 * is where ESBMC already keeps its kernel operational-model headers -- the
 * directory predates this work and is shared with kernel.c. Nothing here
 * tracks 5.15, and nothing needs to be updated when a kernel moves.
 *
 * Real Linux CXL source is verified separately, in regression/cxl-linux/,
 * against a real kernel tree's own headers. No file includes both, so the two
 * cannot drift into disagreement; they simply do not meet. Making them mean
 * the same thing is Phase 7 work, not a versioning question.
 */
#ifndef _LINUX_CXL_H
#define _LINUX_CXL_H

/* size_t, for the CDAT declarations below: this header must stand on its own
 * rather than rely on a translation unit having included <stddef.h> first. */
#include <stddef.h>

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
  unsigned int ways; /* interleave ways; 0 until cxl_region_config() */
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
/* Downstream ports a modelled host bridge may report. The array is always
 * allocated at this size; num_devices says how much of it is populated.
 * Two is enough to exercise "more than one port" -- raising it costs solver
 * time in every test that walks the topology, for no extra behaviour. */
#define CXL_MAX_DOWNSTREAM_PORTS 2

struct cxl_host_bridge {
  struct pci_dev *pdev;
  /* One contiguous array, not an array of separately allocated pointers.
   * The latter costs a distinct heap object per downstream device, and the
   * alias reasoning over a symbolically indexed array of those makes any
   * test that walks the topology unsolvable -- 120s+ for four devices. */
  struct cxl_dev *devices;
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

/* CXL HDM decoder limits (CXL 2.0 §8.2.2.12) */
#define CXL_HDM_DECODER_MAX    8
#define CXL_HDM_ALIGNMENT      4096  /* 4KB minimum alignment */

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

/* ============================================================
 *  PCIe AER (Advanced Error Reporting) — declared here, modelled
 *  in cxl_driver.c
 * ============================================================ */

/* AER error severity levels (PCIe r4.0 §7.10) */
enum aer_error_severity {
  AER_CORRECTABLE    = 0,
  AER_NON_FATAL      = 1,
  AER_FATAL          = 2,
};

int pci_enable_aer(struct pci_dev *dev);
void pci_aer_clear(struct pci_dev *dev, int severity);
int pci_aer_get_first_error(struct pci_dev *dev, int *severity);
int pci_aer_clear_first_error(struct pci_dev *dev);

/* ============================================================
 *  CXL error injection — declared here, modelled in cxl_driver.c
 * ============================================================ */

enum cxl_error_type {
  CXL_ERR_CORRECTABLE = 0,
  CXL_ERR_NON_FATAL,
  CXL_ERR_FATAL,
};

int cxl_err_inject(struct cxl_dev *cxld, enum cxl_error_type type);
int cxl_err_get_count(struct cxl_dev *cxld,
                      int *correctable,
                      int *non_fatal,
                      int *fatal);

/* ============================================================
 *  CXL region interleave — declared here, modelled in cxl_driver.c
 * ============================================================
 *
 * Constraints follow CXL 3.0 §8.2.4.20.1 and the encoders in
 * drivers/cxl/core/region.c (granularity_to_eig(), ways_to_eiw()):
 * interleave granularity is a power of two in [256, 16384] bytes, and
 * interleave ways is a power of two in [1, 16].  The 3/6/12/16-way
 * non-power-of-two encodings are not modelled.
 */

#define CXL_DECODER_MIN_GRANULARITY 256
#define CXL_DECODER_MAX_GRANULARITY 16384
#define CXL_DECODER_MAX_WAYS 16

int cxl_region_config(
  struct cxl_region *region,
  unsigned int ways,
  unsigned int granularity);

/* Non-zero when the two regions share any host physical address. */
int cxl_region_overlaps(const struct cxl_region *a, const struct cxl_region *b);

/* ============================================================
 *  CXL mailbox IOCTL — declared here, modelled in cxl_driver.c
 * ============================================================
 *
 * Models the user-facing path in drivers/cxl/core/mbox.c: cxl_send_cmd()
 * looks the opcode up in cxl_mem_commands[], rejects anything absent with
 * -ENOTTY, rejects a command the device has not enabled
 * (cxlds->enabled_cmds) with the same code, and bounds the payload before
 * touching it.
 */

/* Largest payload the modelled mailbox accepts (CXL 2.0 §8.2.8.4.3). */
#define CXL_MBOX_IOCTL_MAX_PAYLOAD 4096

/* Slot in the modelled command table, or negative when absent. */
int cxl_mbox_cmd_index(u16 opcode);

int cxl_mailbox_ioctl(
  struct cxl_dev *cxld,
  u16 opcode,
  void *payload,
  u32 size);

/* ============================================================
 *  CXL downstream ports (dports) — declared here, modelled in
 *  cxl_driver.c
 * ============================================================
 *
 * drivers/cxl/core/port.c keeps a switch's downstream ports in a list
 * owned by the parent port; Linux spells these devm_cxl_add_dport() and
 * cxl_find_dport_by_dev(). The model keeps the same lifetime rule, so a
 * dport pointer cached across a removal dangles.
 */

#define CXL_PORT_MAX_DPORTS 8

struct cxl_dport
{
  struct cxl_port *port;
  struct pci_dev *dport_dev;
  int id;
};

int cxl_dport_add(struct cxl_port *port, struct pci_dev *dport_dev, int id);
struct cxl_dport *cxl_dport_find(struct cxl_port *port, int id);
void cxl_dport_remove(struct cxl_port *port, int id);
int cxl_dport_count(struct cxl_port *port);

/* ============================================================
 *  ACPI CEDT / CFMWS — declared here, modelled in cxl_driver.c
 * ============================================================
 *
 * Mirrors cxl_acpi_cfmws_verify() in drivers/cxl/acpi.c and eiw_to_ways()
 * in drivers/cxl/cxl.h.  These are the real encodings, not invented ones:
 * the CFMWS interleave-ways field is an *encoded* value (EIW), not a count,
 * and the 256 MB alignment is what the driver actually enforces.
 */

#define CXL_SZ_256M 0x10000000UL

/* acpi_cedt_cfmws.interleave_arithmetic */
#define ACPI_CEDT_CFMWS_ARITHMETIC_MODULO 0
#define ACPI_CEDT_CFMWS_ARITHMETIC_XOR    1

/* Largest ways any valid EIW encodes (EIW 4 -> 16). */
#define CXL_CFMWS_MAX_WAYS 16

struct acpi_cedt_cfmws {
  u32 length;               /* header.length, in bytes */
  u64 base_hpa;
  u64 window_size;
  u8 interleave_ways;       /* EIW encoding, not a count */
  u8 interleave_arithmetic;
  u16 granularity;          /* EIG encoding */
  u32 restrictions;
};

/*
 * CXL ECN "3, 6, 12 and 16-way memory Interleaving": EIW 0..4 encode
 * 1,2,4,8,16 ways; EIW 8..10 encode 3,6,12.  Everything else is invalid.
 * Returns 0 and writes *ways, or -EINVAL.
 */
int eiw_to_ways(u8 eiw, unsigned int *ways);

/*
 * Validates one CFMWS entry exactly as cxl_acpi_cfmws_verify() does.
 * Returns 0, or -EINVAL naming the first violated rule.
 */
int acpi_cedt_parse_cfmws(const struct acpi_cedt_cfmws *cfmws,
                          unsigned int *ways);

/* ============================================================
 *  CDAT (Coherent Device Attribute Table) — declared here,
 *  modelled in cxl_driver.c
 * ============================================================
 *
 * Layouts from include/acpi/actbl1.h; cdat_checksum() is the algorithm in
 * drivers/cxl/core/pci.c and cdat_entry_validate() is the length check every
 * handler in drivers/cxl/core/cdat.c performs before touching an entry.
 *
 * This is the synthetic counterpart to regression/cxl-linux's
 * harness_cdat_checksum pair, which verifies the same property against the
 * real source.
 */

/* enum acpi_cdat_type */
#define ACPI_CDAT_TYPE_DSMAS   0
#define ACPI_CDAT_TYPE_DSLBIS  1
#define ACPI_CDAT_TYPE_DSMSCIS 2
#define ACPI_CDAT_TYPE_DSIS    3
#define ACPI_CDAT_TYPE_DSEMTS  4
#define ACPI_CDAT_TYPE_SSLBIS  5

struct acpi_cdat_header {
  u8 type;
  u8 reserved;
  u16 length;
};

struct acpi_cdat_dsmas {
  u8 dsmad_handle;
  u8 flags;
  u16 reserved;
  u64 dpa_base_address;
  u64 dpa_length;
};

/*
 * Sums the bytes of a CDAT table modulo 256.  A well-formed table sums to
 * zero; any non-zero result means the table is corrupt.  `size` must not
 * exceed the buffer -- that bound is the caller's obligation, and is exactly
 * what read_cdat_data() gets wrong when the DOE read reports back a length
 * larger than the allocation.
 */
unsigned char cdat_checksum(const void *buf, size_t size);

/*
 * Reproduces the guard every cdat_*_handler() runs:
 *   if (len != size || (unsigned long)hdr + len > end) reject
 * Returns 0 if the entry is well-formed and wholly inside the table, or
 * -EINVAL.
 */
int cdat_entry_validate(const struct acpi_cdat_header *hdr,
                        size_t expected_size, const void *end);

/* ============================================================
 *  PCIe DVSEC for CXL Device — declared here, modelled in
 *  cxl_driver.c
 * ============================================================
 *
 * Offsets and field masks from include/uapi/linux/pci_regs.h (CXL r4.0
 * §8.1.3); the enumeration mirrors cxl_dvsec_mem_range_valid() and
 * cxl_dvsec_rr_decode() in drivers/cxl/core/pci.c.
 *
 * Note the shape of the trap: HDM_COUNT is a two-bit field, so the device
 * can report 3, while dvsec_range[] holds CXL_DVSEC_RANGE_MAX == 2. The
 * bound is not implied by the encoding -- the driver has to impose it.
 */

#define PCI_DVSEC_CXL_DEVICE     0
#define PCI_DVSEC_CXL_CAP        0x0A
#define PCI_DVSEC_CXL_MEM_CAPABLE (1U << 2)
#define PCI_DVSEC_CXL_HDM_COUNT_SHIFT 4
#define PCI_DVSEC_CXL_HDM_COUNT_MASK  (0x3U << PCI_DVSEC_CXL_HDM_COUNT_SHIFT)
#define PCI_DVSEC_CXL_CTRL       0x0C
#define PCI_DVSEC_CXL_MEM_ENABLE (1U << 2)

#define PCI_DVSEC_CXL_RANGE_SIZE_HIGH(i) (0x18 + ((i) * 0x10))
#define PCI_DVSEC_CXL_RANGE_SIZE_LOW(i)  (0x1C + ((i) * 0x10))
#define PCI_DVSEC_CXL_MEM_INFO_VALID     (1U << 0)
#define PCI_DVSEC_CXL_MEM_ACTIVE         (1U << 1)
#define PCI_DVSEC_CXL_RANGE_BASE_HIGH(i) (0x20 + ((i) * 0x10))
#define PCI_DVSEC_CXL_RANGE_BASE_LOW(i)  (0x24 + ((i) * 0x10))

#define CXL_DVSEC_RANGE_MAX 2

/* Decodes the HDM range count from a DVSEC capability register value. The
 * result is 0..3 -- the field's full range, not the array's. */
unsigned int cxl_dvsec_hdm_count(u32 cap);

/*
 * Reports whether range `id` has its MEM_INFO_VALID bit set, reading the
 * size-low register at the offset the id dictates.  Returns 1 (valid),
 * 0 (not yet valid) or -EINVAL when id is out of range.
 */
int cxl_dvsec_mem_range_valid(struct pci_dev *pdev, int dvsec, int id);

#endif /* _LINUX_CXL_H */
