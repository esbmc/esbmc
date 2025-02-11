#include <assert.h>
#include <stdint.h>
#include <string.h>

extern struct adj nondet_struct();
extern uintptr_t nondet_uaddr();
struct adj{
  int a;
};


void* buffer_ptr(const unsigned char *buffer, uintptr_t addr){
  return (void*)buffer + ((unsigned char *)addr - buffer);
}

_Bool valid(const unsigned char *buffer, uintptr_t addr, size_t n){
  uintptr_t baddr = (uintptr_t)buffer;
  if(addr % 4 == 0 && baddr <= addr && addr < baddr + n)
    return 1;
  return 0;
}

int main() {
  struct adj adj = nondet_struct();
  adj.a = 2;
  unsigned char buffer[1024] __attribute__((__aligned__(4)));
  (void)memcpy(buffer, &adj, sizeof(adj));
  uintptr_t addr = nondet_uaddr();
  if(!valid(buffer, addr, sizeof(buffer)))
    return 0;
  struct adj* adj2 = buffer_ptr(buffer, addr);
  int v = adj2->a;
  assert(v <= 2); // fails, because buffer is not zero-initialized
}
