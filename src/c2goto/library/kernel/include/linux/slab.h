
#include <linux/gfp.h>
#include <linux/overflow.h>
#include <linux/types.h>
#include <linux/workqueue.h>
#include <linux/percpu-refcount.h>
#include <assert.h>


void check_gfp_flags(gfp_t flags) {
    // Define all valid flags
    gfp_t valid_flags = __GFP_DMA | __GFP_HIGHMEM | __GFP_DMA32 | __GFP_ZERO | __GFP_NOWARN |
                        __GFP_REPEAT | __GFP_NOFAIL | __GFP_NORETRY | __GFP_MEMALLOC | __GFP_COMP |
                        __GFP_NO_KSWAPD | __GFP_OTHER_NODE | __GFP_WRITE | __GFP_HARDWALL |
                        __GFP_THISNODE | __GFP_ATOMIC | __GFP_ACCOUNT | __GFP_RECLAIM | __GFP_IO |
                        __GFP_FS |  GFP_KERNEL | GFP_KERNEL_ACCOUNT | 
                        GFP_NOIO | GFP_NOFS | GFP_USER | GFP_DMA | GFP_DMA32 | GFP_HIGHUSER ;

    // Check if any flag is set that is not in the list of valid flags
    assert((flags & ~valid_flags) == 0);
}
void*__kmalloc(size_t size, gfp_t flags)
{
    return malloc(size);
}

void *__kmalloc_large(size_t size, gfp_t flags) {
    (void)flags;  // Ignore flags.
    return malloc(size);
}

void *kmalloc(int size, int flags)
{

    // Check size greater than  zero and less than max
    assert(size > 0 && size <= MAX_ALLOC_SIZE);
    //check flags greater than zero
    assert(flags > 0);

    //check if flags have corresponding valid values
    check_gfp_flags(flags);
    // If the size is larger than the KMALLOC_MAX_CACHE_SIZE, then handle in kmalloc_large
    if (size > KMALLOC_MAX_CACHE_SIZE) {
        // Call to kmalloc_large or equivalent function can be here.
		return __kmalloc_large(size, flags);
    }

	(void)flags;  // Ignore flags.
    return malloc(size);
}


void kfree(const void *ptr) {
    free((void *)ptr);
}
