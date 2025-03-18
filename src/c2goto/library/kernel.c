#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <errno.h>
#include <stdbool.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/spinlock.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/asm/uaccess.h>
#include <assert.h>
#define spin_limit 80

typedef unsigned int gfp_t;

char user_memory[USER_MEMORY_SPACE];     //mock user memory
char kernel_memory[KERNEL_MEMORY_SPACE]; //mock user memory

static void check_gfp_flags(gfp_t flags)
{
__ESBMC_HIDE:;
  // Define all valid flags
  gfp_t valid_flags =
    __GFP_DMA | __GFP_HIGHMEM | __GFP_DMA32 | __GFP_ZERO | __GFP_NOWARN |
    __GFP_REPEAT | __GFP_NOFAIL | __GFP_NORETRY | __GFP_MEMALLOC | __GFP_COMP |
    __GFP_NO_KSWAPD | __GFP_OTHER_NODE | __GFP_WRITE | __GFP_HARDWALL |
    __GFP_THISNODE | __GFP_ATOMIC | __GFP_ACCOUNT | __GFP_RECLAIM | __GFP_IO |
    __GFP_FS | GFP_KERNEL | GFP_KERNEL_ACCOUNT | GFP_NOIO | GFP_NOFS |
    GFP_USER | GFP_DMA | GFP_DMA32 | GFP_HIGHUSER;

  // Check if any flag is set that is not in the list of valid flags
  assert((flags & ~valid_flags) == 0);
}
static void *__kmalloc(size_t size, gfp_t flags)
{
__ESBMC_HIDE:;
  return malloc(size);
}

static void *__kmalloc_large(size_t size, gfp_t flags)
{
__ESBMC_HIDE:;
  (void)flags; // Ignore flags.
  return malloc(size);
}

void *kmalloc(int size, int flags)
{
__ESBMC_HIDE:;
  // Check size greater than  zero and less than max
  assert(size > 0 && size <= MAX_ALLOC_SIZE);
  //check flags greater than zero
  assert(flags > 0);

  //check if flags have corresponding valid values
  check_gfp_flags(flags);
  // If the size is larger than the KMALLOC_MAX_CACHE_SIZE, then handle in kmalloc_large
  if (size > KMALLOC_MAX_CACHE_SIZE)
  {
    // Call to kmalloc_large or equivalent function can be here.
    return __kmalloc_large(size, flags);
  }

  (void)flags; // Ignore flags.
  return __kmalloc(size, flags);
}

void kfree(const void *ptr)
{
__ESBMC_HIDE:;
  free((void *)ptr);
}
void *kmalloc_array(size_t n, size_t size, gfp_t flags)
{
__ESBMC_HIDE:;
  return __kmalloc(n * size, flags);
}

void *kcalloc(size_t n, size_t size, gfp_t flags)
{
__ESBMC_HIDE:;
  (void)flags;
  return calloc(n, size);
}

unsigned long copy_to_user(void *to, void *from, unsigned long size)
{
__ESBMC_HIDE:;
  //checking on the passed parameters of kernel function
  //the source in kernel space and destination in user space must be valid
  assert(to != NULL);
  assert(from != NULL);
  assert(size <= PAGE_SIZE);

  assert((char *)to >= user_memory);
  assert((char *)from >= kernel_memory);

  //copy memory from kernel space to user space
  //simulate the copy operation by memcpy
  memcpy(to, from, size);

  return 0;
}

unsigned long copy_from_user(void *to, void *from, unsigned long size)
{
__ESBMC_HIDE:;
  //the source in user space and destination in kernel space must be valid
  //avoid dereferencing null pointer
  assert(to != NULL);
  assert(from != NULL);
  assert(size <= PAGE_SIZE);

  assert((char *)to >= kernel_memory);
  assert((char *)from >= user_memory);
  //copy memory from user space to kernel space
  //simulate the copy operation by memcpy
  memcpy(to, from, size);

  return 0;
}

void spin_lock_init(spinlock_t *lock)
{
__ESBMC_HIDE:;
  //check if the lock is valid
  assert(lock != NULL);
  //initialize the lock
  lock->locked = false;
}

bool spin_lock(spinlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_assert(lock != NULL, "The lock is null, verfication failed");

  int retries = 0;
  while (retries < SPIN_LIMIT)
  {
    __ESBMC_atomic_begin();
    if (lock->locked == false)
    {
      lock->locked = true;
      __ESBMC_atomic_end();
      return true;
    }
    __ESBMC_atomic_end();
    retries++;
  }

  return false;
}

void spin_unlock(spinlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_assert(lock != NULL, "The lock is null, verfication failed");
  lock->locked = false;
}
