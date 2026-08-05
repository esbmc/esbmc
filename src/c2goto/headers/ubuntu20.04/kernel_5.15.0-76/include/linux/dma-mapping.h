/* SPDX-License-Identifier: GPL-2.0
 *
 * linux/dma-mapping.h — DMA API header (ESBMC operational model stubs).
 *
 * This header provides declarations for the DMA (Direct Memory Access) API
 * used by device drivers. Function bodies are modeled in
 * src/c2goto/library/cxl_driver.c.
 */
#ifndef _LINUX_DMA_MAPPING_H
#define _LINUX_DMA_MAPPING_H

#include <stdint.h>
#include <stddef.h>

/* Device type (minimal stub) — must be before function declarations */
struct device {
  const char *init_name;
  void *driver_data;
};

/* DMA direction */
enum dma_data_direction {
  DMA_BIDIRECTIONAL = 0,
  DMA_TO_DEVICE     = 1,
  DMA_FROM_DEVICE   = 2,
  DMA_NONE          = 3,
};

/* DMA mapping handle — represents a device-visible address */
typedef uint64_t dma_addr_t;

/* ============================================================
 *  DMA API — declared here, modelled in cxl_driver.c
 * ============================================================ */

/* Coherent (non-streaming) DMA allocation */
void *dma_alloc_coherent(struct device *dev, size_t size,
                         dma_addr_t *dma_handle, unsigned int flag);
void dma_free_coherent(struct device *dev, size_t size,
                       void *cpu_addr, dma_addr_t dma_handle);

/* Streaming DMA mapping */
dma_addr_t dma_map_single(struct device *dev, void *cpu_addr, size_t size,
                          enum dma_data_direction dir);
void dma_unmap_single(struct device *dev, dma_addr_t dma_handle, size_t size,
                      enum dma_data_direction dir);

/* DMA sync for CPU/device */
void dma_sync_single_for_cpu(struct device *dev, dma_addr_t dma_handle,
                             size_t size, enum dma_data_direction dir);
void dma_sync_single_for_device(struct device *dev, dma_addr_t dma_handle,
                                size_t size, enum dma_data_direction dir);

/* DMA mapping attributes */
int dma_set_mask(struct device *dev, uint64_t mask);
int dma_set_coherent_mask(struct device *dev, uint64_t mask);

#endif /* _LINUX_DMA_MAPPING_H */
