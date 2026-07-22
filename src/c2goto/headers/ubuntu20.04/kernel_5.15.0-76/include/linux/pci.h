/* SPDX-License-Identifier: GPL-2.0
 *
 * linux/pci.h — PCI subsystem header (ESBMC operational model stubs).
 *
 * This header provides declarations for the PCI subsystem used by CXL
 * device drivers. Function bodies are modeled in src/c2goto/library/cxl_driver.c.
 */
#ifndef _LINUX_PCI_H
#define _LINUX_PCI_H

#include <ubuntu20.04/kernel_5.15.0-76/include/asm/io.h>

/* Basic kernel types */
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long resource_size_t;

/* PCI BAR (Base Address Register) */
#define PCI_NUM_RESOURCES     6
#define PCI_ROM_RESOURCE      6
#define PCI_NUM_RESOURCES_WITH_ROM 7

/* BAR type bits */
#define PCI_BASE_ADDRESS_SPACE_MEMORY  0
#define PCI_BASE_ADDRESS_SPACE_IO      1
#define PCI_BASE_ADDRESS_MEM_TYPE_32   0
#define PCI_BASE_ADDRESS_MEM_TYPE_64   1
#define PCI_BASE_ADDRESS_MEM_PREFETCH  0x08

/* PCI command register bits */
#define PCI_COMMAND_IO         0x01
#define PCI_COMMAND_MEMORY     0x02
#define PCI_COMMAND_MASTER     0x04
#define PCI_COMMAND_SERR       0x08
#define PCI_COMMAND_INTX_DISABLE 0x080

/* PCI latency timer */
#define PCI_DEFAULT_LATENCY_TIMER  16

/* PCI subsystem */
struct pci_bus {
  u32 domain;
  u8  busnr;
  struct pci_dev *devices;
};

/* PCI device */
struct pci_dev {
  u16 vendor;
  u16 device;
  u16 subsystem_vendor;
  u16 subsystem_device;
  u16 class_;
  u8  revision;
  u8  hdr_type;
  u8  pin;        /* interrupt pin (INTA-INTD) */

  /* BARs */
  resource_size_t resource_start[PCI_NUM_RESOURCES_WITH_ROM];
  resource_size_t resource_size[PCI_NUM_RESOURCES_WITH_ROM];
  u32 resource_flags[PCI_NUM_RESOURCES_WITH_ROM];

  /* Interrupt */
  u32 irq;
  struct pci_bus *bus;

  /* Driver model */
  void *driver_data;
  struct pci_driver *driver;
};

/* PCI driver */
struct pci_driver {
  const char *name;
  const struct pci_device_id *id_table;
  int (*probe)(struct pci_dev *dev, const struct pci_device_id *id);
  void (*remove)(struct pci_dev *dev);
};

/* PCI device ID table */
struct pci_device_id {
  u16 vendor, device;
  u16 subvendor, subdevice;
  u32 class_, class_mask;
  unsigned long driver_data;
};

/* PCI configuration space access */
#define PCI_CFG_SPACE_SIZE     256
#define PCI_CFG_SPACE_EXP_SIZE 4096

/* ============================================================
 *  PCI API — declared here, modelled in cxl_driver.c
 * ============================================================ */

/* Device enumeration */
struct pci_dev *pci_get_device(u16 vendor, u16 device, struct pci_dev *from);
struct pci_dev *pci_get_bus_device(u32 domain, u8 bus, u8 devfn);
void pci_put_device(struct pci_dev *dev);

/* BAR management */
int pci_enable_device(struct pci_dev *dev);
void pci_disable_device(struct pci_dev *dev);
int pci_request_regions(struct pci_dev *dev, const char *res_name);
void pci_release_regions(struct pci_dev *dev);
resource_size_t pci_resource_start(struct pci_dev *dev, int bar);
resource_size_t pci_resource_end(struct pci_dev *dev, int bar);
u32 pci_resource_flags(struct pci_dev *dev, int bar);

/* MMIO mapping */
void *pci_iomap(struct pci_dev *dev, int bar, unsigned long max);
void pci_iounmap(struct pci_dev *dev, void *addr);

/* Interrupt */
int pci_enable_msi(struct pci_dev *dev);
void pci_disable_msi(struct pci_dev *dev);
int pci_alloc_irq_vectors(struct pci_dev *dev, unsigned int min_vecs,
                          unsigned int max_vecs, unsigned int flags);
void pci_free_irq_vectors(struct pci_dev *dev);

/* Driver registration */
int pci_register_driver(struct pci_driver *drv);
void pci_unregister_driver(struct pci_driver *drv);

/* Configuration space access */
int pci_read_config_byte(struct pci_dev *dev, int where, u8 *val);
int pci_read_config_word(struct pci_dev *dev, int where, u16 *val);
int pci_read_config_dword(struct pci_dev *dev, int where, u32 *val);
int pci_write_config_byte(struct pci_dev *dev, int where, u8 val);
int pci_write_config_word(struct pci_dev *dev, int where, u16 val);
int pci_write_config_dword(struct pci_dev *dev, int where, u32 val);

#endif /* _LINUX_PCI_H */
