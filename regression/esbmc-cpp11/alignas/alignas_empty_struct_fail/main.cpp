#include <assert.h>
#include <string.h>

// Negative counterpart of alignas_empty_struct: the over-aligned empty struct
// now occupies its full 16-byte alignment and `= {}` zero-initialises every
// byte, so comparing it to a 16-byte zero buffer is equal. Asserting they
// differ must fail -- pinning that the padded bytes are readable and zero.
struct alignas(16) aligned_empty
{
};

int main()
{
  struct aligned_empty test = {};
  char zeroes[16] = {};
  assert(memcmp(&test, &zeroes, 16) != 0);
  return 0;
}
