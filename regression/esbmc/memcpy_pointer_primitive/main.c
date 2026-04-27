// Exercises gen_byte_memcpy's pointer-type early return in
// src/goto-symex/builtin_functions/memory_ops.cpp (line 365-366):
// memcpy between two pointer variables of *different* pointer types
// skips the short-circuit and falls into gen_byte_memcpy, which then
// returns nil for pointer operands. intrinsic_memcpy handles this by
// bumping the call to __memcpy_impl.
#include <assert.h>
#include <string.h>

int main()
{
  int *a = (int *)0;
  long *b = (long *)0;

  memcpy(&a, &b, sizeof(a));

  // After a byte-wise copy, a holds the same bit pattern as b — NULL.
  assert(a == (int *)0);
  return 0;
}
