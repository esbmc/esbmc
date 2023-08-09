/* SPDX-License-Identifier: GPL-2.0 */
#include <stdatomic.h>
/* The declarations of mock spin lock behaviour*/
/** mock spinlock struct*/
typedef struct {
    int locked;
} spinlock_t;

void spin_lock_init(spinlock_t *lock);

void spin_lock(spinlock_t *lock);

void spin_unlock(spinlock_t *lock);