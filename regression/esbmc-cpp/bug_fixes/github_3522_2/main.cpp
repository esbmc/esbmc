#include <assert.h>
#include <cstdint>
#include <string.h>

struct __attribute__((__aligned__)) __aligned_storage_max_align_t { char a; };
struct __aligned_storage_max_align_t2 {};

const long align1 = alignof(__aligned_storage_max_align_t);
const long sizeof1 = sizeof(__aligned_storage_max_align_t);
const long align2 = alignof(__aligned_storage_max_align_t2);
const long sizeof2 = sizeof(__aligned_storage_max_align_t2);

int main() {
  struct __aligned_storage_max_align_t test = {};
  const char zeroes[sizeof1] = {};
  assert(align1 == 16L);
  assert(sizeof1 == 16L);
  assert(align2 == 1L);
  assert(sizeof2 == 1L);
  assert(!memcmp(&test, &zeroes, sizeof1));
  assert((uintptr_t)&test % align1 == 0);
}