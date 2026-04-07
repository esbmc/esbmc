// Test: struct with multiple unnamed bitfields interleaved with regular
// byte-aligned fields.  Verifies that the sub-byte skip guard does not
// prevent correct access to the byte-aligned fields through a pointer.

#include <assert.h>

struct mixed {
  int            a;        // byte-aligned (offset 0)
  struct { char : 1; };    // anonymous sub-struct with 1-bit unnamed bitfield
  long           b;        // byte-aligned
  struct { char : 3; };    // anonymous sub-struct with 3-bit unnamed bitfield
  short          c;        // byte-aligned
};

int main(void) {
  struct mixed m;
  m.a = 100;
  m.b = 200;
  m.c = 300;

  struct mixed *p = &m;

  // All three byte-aligned fields must still be accessible and correct
  // despite the presence of sub-byte padding members in between.
  assert(p->a == 100);
  assert(p->b == 200);
  assert(p->c == 300);

  return 0;
}
