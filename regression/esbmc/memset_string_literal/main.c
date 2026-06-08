// Exercises intrinsic_memset's read-only-target detection in
// src/goto-symex/builtin_functions/memory_ops.cpp. When memset would
// write into a string literal, the optimiser must bail out and let
// __memset_impl report the violation through WRITE-mode dereference.
#include <string.h>

int main()
{
  char *s = "hello";
  // Writing through a pointer into a string literal is undefined
  // behaviour; ESBMC should detect this and fail verification.
  memset(s, 'x', 1);
  return 0;
}
