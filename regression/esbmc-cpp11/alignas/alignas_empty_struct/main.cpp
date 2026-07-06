#include <assert.h>
#include <string.h>

// An over-aligned empty struct occupies its alignment (16 bytes here), so its
// byte layout must include trailing padding to match sizeof; otherwise the
// memcmp below reads out of bounds.
struct alignas(16) __aligned_storage_max_align_t
{
};
struct __aligned_storage_max_align_t2
{
};

const long align1 = alignof(__aligned_storage_max_align_t);
const long sizeof1 = sizeof(__aligned_storage_max_align_t);
const long align2 = alignof(__aligned_storage_max_align_t2);
const long sizeof2 = sizeof(__aligned_storage_max_align_t2);

int main()
{
  struct __aligned_storage_max_align_t test = {};
  char zeroes[sizeof1] = {};
  assert(align1 == 16L);
  assert(sizeof1 == 16L);
  assert(align2 == 1L);
  assert(sizeof2 == 1L);
  assert(!memcmp(&test, &zeroes, sizeof1));
}