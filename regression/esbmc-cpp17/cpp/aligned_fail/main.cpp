#include <iostream>
#include <new> // For std::aligned_alloc
#include <cassert>
#include <cstdlib>

int main()
{
  size_t alignment = 16;
  size_t size = 128;

  void *ptr =
    std::aligned_alloc(static_cast<size_t>(std::align_val_t(alignment)), size);

  assert(ptr == nullptr);

  free(ptr);

  return 0;
}