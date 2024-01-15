#include <assert.h>
#include <stdint.h>
#include <string.h>

unsigned char buffer[1024] __attribute__((__aligned__(4)));
extern struct adj nondet_struct();
extern uintptr_t nondet_uaddr();
struct adj{
  int a;
};


void* buffer_ptr(uintptr_t addr){
  return (void*)buffer + ((unsigned char *)addr - buffer);
}

_Bool valid(uintptr_t addr){
  if(addr % 4 == 0 && (uintptr_t)buffer <= addr && addr < (uintptr_t)buffer + sizeof(buffer))
    return 1;
  return 0;
}

int main() {
  struct adj adj = nondet_struct();
  adj.a = 2;
  (void)memcpy(buffer, &adj, sizeof(adj));
  uintptr_t addr = nondet_uaddr();
  if(!valid(addr))
    return 0;
  struct adj* adj2 = buffer_ptr(addr);
  assert(adj2->a <= 2);
}
