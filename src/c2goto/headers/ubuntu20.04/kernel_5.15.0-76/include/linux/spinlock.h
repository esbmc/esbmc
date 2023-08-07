/* SPDX-License-Identifier: GPL-2.0 */
#include <stdatomic.h>
/* The declarations of mock spin lock behaviour*/
/** mock spinlock struct*/
typedef struct {
    int locked;
} mock_spinlock_t;

void kernel_spinlock_init(mock_spinlock_t *lock);

void kernel_spin_lock(mock_spinlock_t *lock);

void kernel_spin_unlock(mock_spinlock_t *lock);