/* SPDX-License-Identifier: GPL-2.0
 *
 * linux/irq.h — Interrupt handling header (ESBMC operational model stubs).
 *
 * This header provides declarations for interrupt handling used by
 * device drivers. Function bodies are modeled in
 * src/c2goto/library/cxl_driver.c.
 */
#ifndef _LINUX_IRQ_H
#define _LINUX_IRQ_H

#include <stdint.h>
#include <stdbool.h>

/* IRQ return codes */
#define IRQ_NONE    (0)
#define IRQ_HANDLED (1)
#define IRQ_WAKE_THREAD (2)

/* IRQ flag bits */
#define IRQF_SHARED       0x00000080
#define IRQF_ONESHOT      0x00010000
#define IRQF_TRIGGER_NONE   0x00000000
#define IRQF_TRIGGER_RISING 0x00000001
#define IRQF_TRIGGER_FALLING 0x00000002
#define IRQF_TRIGGER_HIGH   0x00000004
#define IRQF_TRIGGER_LOW    0x00000008

/* Interrupt handler type */
typedef void (*irq_handler_t)(int irq, void *dev_id);

/* ============================================================
 *  IRQ API — declared here, modelled in cxl_driver.c
 * ============================================================ */

/* Request/free IRQ */
int request_irq(unsigned int irq, irq_handler_t handler, unsigned long flags,
                const char *name, void *dev_id);
void free_irq(unsigned int irq, void *dev_id);

/* Interrupt enable/disable */
void disable_irq(unsigned int irq);
void enable_irq(unsigned int irq);
void disable_irq_nosync(unsigned int irq);
void synchronize_irq(unsigned int irq);

/* Interrupt masking */
void mask_irq(unsigned int irq);
void unmask_irq(unsigned int irq);

/* Simulate an interrupt firing (for testing models) */
void esbmc_simulate_irq(unsigned int irq, void *dev_id);

#endif /* _LINUX_IRQ_H */
