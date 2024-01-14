#include <assert.h>
#include <string.h>

unsigned char buffer[1024] __attribute__((__aligned__(4)));
extern struct adj nondet_struct();
extern unsigned long nondet_ulong();
struct adj{
  int a;
};


void* buffer_ptr(unsigned long addr){
  return (void*)buffer + ((unsigned char *)addr - buffer);
}

_Bool valid(unsigned long addr){
  if(addr % 4 == 0 && (unsigned long)buffer <= addr && addr < (unsigned long)buffer + sizeof(buffer))
    return 1;
  return 0;
}

int main() {
  struct adj adj = nondet_struct();
  adj.a = 2;
  (void)memcpy(buffer, &adj, sizeof(adj));
  unsigned long addr = nondet_ulong();
  if(!valid(addr))
    return 0;
  struct adj* adj2 = buffer_ptr(addr);
  assert(adj2->a <= 2);
}
