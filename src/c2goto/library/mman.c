#include <stdlib.h>
#include <sys/mman.h>

#undef mmap
#undef munmap

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
__ESBMC_HIDE:;
  (void *)addr;
  (void)prot;
  (void)flags;
  (void)fd;
  (void)offset;
  void *res = malloc(length);
  if(!res)
    return MAP_FAILED;
  else
    return res;
}

int munmap(void *addr, size_t length)
{
__ESBMC_HIDE:;
  (void)length;
  if(addr)
  {
    free(addr);
    return 0;
  }
  else
    return -1;
}
