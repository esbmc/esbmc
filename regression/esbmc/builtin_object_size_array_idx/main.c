// Exercises intrinsic_builtin_object_size's index2t branch in
// src/goto-symex/engine/builtin_functions/object_size.cpp:
// when the argument is &arr[i], the address_of wraps an index2t whose
// source_value is a symbol (the array) — the size is read from that
// source's array type.
#include <assert.h>
#include <stddef.h>

int main()
{
  int a[10];
  size_t s = __builtin_object_size(&a[2], 0);

  // Type 0 counts from the pointer to the end of the whole array (#8660).
  assert(s == 8 * sizeof(int));
  return 0;
}
