/* SPDX-License-Identifier: GPL-2.0
 *
 * asm/io.h — MMIO access header (ESBMC operational model stubs).
 *
 * This header provides declarations for MMIO (Memory-Mapped I/O) access
 * functions used by device drivers. Function bodies are modeled in
 * src/c2goto/library/cxl_driver.c.
 */
#ifndef _ASM_IO_H
#define _ASM_IO_H

#include <stdint.h>
#include <stdbool.h>

/* ============================================================
 *  MMIO read functions — declared here, modelled in cxl_driver.c
 * ============================================================ */

/* 8-bit reads */
uint8_t  readb(const void *addr);
uint16_t readw(const void *addr);
uint32_t readl(const void *addr);
uint64_t readq(const void *addr);

/* 8-bit writes */
void writeb(uint8_t  val, const void *addr);
void writew(uint16_t val, const void *addr);
void writel(uint32_t val, const void *addr);
void writeq(uint64_t val, const void *addr);

/* Block reads/writes */
void readsl(const void *addr, void *buf, unsigned long count);
void writesl(const void *addr, const void *buf, unsigned long count);
void readsw(const void *addr, void *buf, unsigned long count);
void writesw(const void *addr, const void *buf, unsigned long count);

/* ============================================================
 *  I/O port functions (less common for CXL, but included for
 *  completeness with the kernel header set)
 * ============================================================ */

uint8_t  inb(unsigned long port);
uint16_t inw(unsigned long port);
uint32_t inl(unsigned long port);
void outb(uint8_t  val, unsigned long port);
void outw(uint16_t val, unsigned long port);
void outl(uint32_t val, unsigned long port);

/* ============================================================
 *  Memory barriers (critical for MMIO ordering)
 * ============================================================ */

/* Full memory barrier */
void mb(void);

/* Write barrier (ensures prior writes reach the device) */
void wmb(void);

/* Read barrier (ensures prior reads complete before subsequent ones) */
void rmb(void);

/* Smp-specific variants */
void smp_mb(void);
void smp_wmb(void);
void smp_rmb(void);

/* Readl_relaxed / writel_relaxed — no ordering guarantee */
uint32_t readl_relaxed(const void *addr);
void writel_relaxed(uint32_t val, const void *addr);

#endif /* _ASM_IO_H */
