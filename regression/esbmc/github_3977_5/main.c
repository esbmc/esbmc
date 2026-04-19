// Test: constant-offset access triggering construct_from_const_struct_offset
// on a struct where multiple unnamed bitfields precede a regular field,
// and the struct is accessed through pointer arithmetic with a known offset.

#include <assert.h>
#include <stddef.h>

typedef struct {
  struct { char : 1; };    // 1-bit unnamed bitfield (sub-byte)
  struct { char : 1; };    // another 1-bit unnamed bitfield
  int data;
} padded;

typedef struct {
  padded items[2];
} container;

int main(void) {
  container c;
  c.items[0].data = 111;
  c.items[1].data = 222;

  // Constant-offset access via array index (known at compile time)
  container *p = &c;
  assert(p->items[0].data == 111);
  assert(p->items[1].data == 222);

  return 0;
}
