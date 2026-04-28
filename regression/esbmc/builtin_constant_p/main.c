// Exercises run_builtin's __builtin_constant_p handler in
// src/goto-symex/builtin_functions/run_builtin.cpp. The handler
// renames the operand and assigns 1 to ret when the renamed value
// is a constant_int2t, 0 otherwise.
#include <assert.h>

int main()
{
  // A compile-time constant literal: after renaming, the operand is
  // a constant_int2t, so __builtin_constant_p returns 1.
  assert(__builtin_constant_p(42));
  return 0;
}
