/* SPDX-License-Identifier: GPL-2.0 */
#include <stdatomic.h>
#include <stdbool.h>
/* The declarations of mock spin lock behavior*/
/** mock spinlock struct*/
#define SPIN_LIMIT 80
typedef struct {
    bool locked;
} spinlock_t;

void spin_lock_init(spinlock_t *lock);

bool spin_lock(spinlock_t *lock);

void spin_unlock(spinlock_t *lock);
