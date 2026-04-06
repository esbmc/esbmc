// Test: dynamic-offset access into an array of structs that contain
// unnamed bitfields alongside regular fields.
// Exercises construct_from_dyn_struct_offset: the sub-byte members
// must be skipped in the if-then-else chain while byte-aligned fields
// remain correctly accessible.

#include <assert.h>

typedef struct {
  struct { char : 1; };    // sub-byte unnamed bitfield
  int key;
  struct { char : 2; };    // another sub-byte unnamed bitfield
  int val;
} entry;

int main(void) {
  entry arr[3];
  arr[0].key = 10;  arr[0].val = 100;
  arr[1].key = 20;  arr[1].val = 200;
  arr[2].key = 30;  arr[2].val = 300;

  // Dynamic (nondet) index — forces dynamic-offset dereference path
  int i;
  __ESBMC_assume(i >= 0 && i < 3);

  entry *p = &arr[i];

  // Verify byte-aligned field access through dynamic offset still works
  assert(p->key >= 10 && p->key <= 30);
  assert(p->val >= 100 && p->val <= 300);

  return 0;
}
