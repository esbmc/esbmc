/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Written by Mark Hemment, 1996 (markhe@nextd.demon.co.uk).
 *
 * (C) SGI 2006, Christoph Lameter
 * 	Cleaned up and restructured to ease the addition of alternative
 * 	implementations of SLAB allocators.
 * (C) Linux Foundation 2008-2013
 *      Unified interface for all slab allocators
 */

// TODO: Replace hardcoded MAX_ALLOC_SIZE with a dynamically obtained value
#define MAX_ALLOC_SIZE 1024*1024 // Maximum allocatable size set to 1MB for example


#define ZERO_SIZE_PTR ((void *)16)

/* Maximum allocatable size */
#define KMALLOC_MAX_SIZE	(1UL << KMALLOC_SHIFT_MAX)
/* Maximum size for which we actually use a slab cache */
#define KMALLOC_MAX_CACHE_SIZE	1024
/* Maximum order allocatable via the slab allocator */
#define KMALLOC_MAX_ORDER	(KMALLOC_SHIFT_MAX - PAGE_SHIFT)

/*
 * Kmalloc subsystem.
 */
#ifndef KMALLOC_MIN_SIZE
#define KMALLOC_MIN_SIZE (1 << KMALLOC_SHIFT_LOW)
#endif



#define __GFP_DMA         0x01u
#define __GFP_HIGHMEM     0x02u
#define __GFP_DMA32       0x04u
#define __GFP_ZERO        0x40u
#define __GFP_NOWARN      0x80u
#define __GFP_REPEAT      0x400u
#define __GFP_NOFAIL      0x800u
#define __GFP_NORETRY     0x1000u
#define __GFP_MEMALLOC    0x2000u
#define __GFP_COMP        0x4000u
#define __GFP_NO_KSWAPD   0x8000u
#define __GFP_OTHER_NODE  0x10000u
#define __GFP_WRITE       0x20000u
#define __GFP_HARDWALL    0x40000u
#define __GFP_THISNODE    0x80000u
#define __GFP_ATOMIC      0x100000u
#define __GFP_ACCOUNT     0x200000u
#define __GFP_RECLAIM 0x01u
#define __GFP_IO      0x02u
#define __GFP_FS      0x04u

#define GFP_ATOMIC	(__GFP_HIGH|__GFP_ATOMIC|__GFP_KSWAPD_RECLAIM)
#define GFP_KERNEL	(__GFP_RECLAIM | __GFP_IO | __GFP_FS)
#define GFP_KERNEL_ACCOUNT (GFP_KERNEL | __GFP_ACCOUNT)
#define GFP_NOWAIT	(__GFP_KSWAPD_RECLAIM)
#define GFP_NOIO	(__GFP_RECLAIM)
#define GFP_NOFS	(__GFP_RECLAIM | __GFP_IO)
#define GFP_USER	(__GFP_RECLAIM | __GFP_IO | __GFP_FS | __GFP_HARDWALL)
#define GFP_DMA		__GFP_DMA
#define GFP_DMA32	__GFP_DMA32
#define GFP_HIGHUSER	(GFP_USER | __GFP_HIGHMEM)
#define GFP_HIGHUSER_MOVABLE	(GFP_HIGHUSER | __GFP_MOVABLE | \
			 __GFP_SKIP_KASAN_POISON)
#define GFP_TRANSHUGE_LIGHT	((GFP_HIGHUSER_MOVABLE | __GFP_COMP | \
			 __GFP_NOMEMALLOC | __GFP_NOWARN) & ~__GFP_RECLAIM)
#define GFP_TRANSHUGE	(GFP_TRANSHUGE_LIGHT | __GFP_DIRECT_RECLAIM)

/* Convert GFP flags to their corresponding migrate type */
#define GFP_MOVABLE_MASK (__GFP_RECLAIMABLE|__GFP_MOVABLE)
#define GFP_MOVABLE_SHIFT 3

// Define size_t if it's not already defined. This is often
// done in stddef.h or sys/types.h, so you might #include one of those instead.

// Define gfp_t and a few example flags.
typedef unsigned int gfp_t;
