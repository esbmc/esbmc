// Test: constant-offset pointer dereference into a struct containing
// an anonymous sub-struct with an unnamed 1-bit bitfield.
// Exercises construct_from_const_struct_offset sub-byte skip guard.

#include <assert.h>

typedef struct {
  struct {
    char : 1;           // unnamed bitfield → sub-byte padding member
  };
  int value;
} S;

int main(void) {
  S s;
  s.value = 42;

  S *p = &s;

  // Constant-offset access: the compiler knows the offset of 'value'
  // within S at compile time.  The sub-byte padding member must be
  // skipped so that the dereference of p->value succeeds.
  assert(p->value == 42);

  return 0;
}
