#include <stdint.h>

/* The base-alignment bump is only as large as an in-bounds access to the
 * object could demand, so it must not claim more than the object's size
 * allows: a one-byte object stays 1-aligned. Types that decline alignment opt
 * out entirely, including through an array -- ns.follow() resolves symbol
 * types only, so the subtype has to be reached explicitly. */

struct __attribute__((packed)) packed_s
{
  char c[3];
};

struct packed_s g_packed_array[4];
char g_char;

int main(void)
{
  __ESBMC_assert(((uintptr_t)&g_char % 16) == 0, "one-byte object is 16-aligned");
  __ESBMC_assert(
    ((uintptr_t)g_packed_array % 16) == 0, "array of packed is 16-aligned");
  return 0;
}
