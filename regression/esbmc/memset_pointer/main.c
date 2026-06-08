// Exercises gen_byte_expression's pointer-type branch in
// src/goto-symex/builtin_functions/memory_ops.cpp, which dispatches
// to gen_byte_expression_byte_update for memset on a pointer variable.
#include <assert.h>
#include <stdint.h>
#include <string.h>

int main()
{
  int *p = (int *)0x12345678;

  // memset zeroes the storage of the pointer itself — the target
  // type is (int *), so the pointer-type branch fires.
  memset(&p, 0, sizeof(p));

  assert(p == (int *)0);
  return 0;
}
