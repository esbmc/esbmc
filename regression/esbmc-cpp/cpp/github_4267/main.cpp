// https://github.com/esbmc/esbmc/issues/4267
//
// A misaligned pointer to a __attribute__((packed)) struct must NOT
// produce an "Incorrect alignment when accessing data object" VCC:
// the packed attribute means the access is deliberately unaligned and
// dereference_expr_nonscalar() already marks the access as such. The
// secondary alignment check on the raw pointer dereference must
// respect that signal.
#include <cstdint>

struct [[gnu::packed]] S
{
  uint8_t a;
  uint32_t b;
};

int main()
{
  alignas(4) char buf[16];
  // Place an S so its 4-byte 'b' lands at byte offset 2 in buf,
  // which is not a 4-byte boundary. Because S is packed this is OK.
  S *p = reinterpret_cast<S *>(buf + 1);
  p->b = 0;
  return 0;
}
