#include <cstdint>
#include <climits>

static uint64_t read_mem_spec_gigabytes_unsafe(uint64_t parsed_value)
{
  const uint64_t mult = 1024ULL * 1024ULL * 1024ULL;
  return parsed_value * mult;
}

int main(void)
{
  uint64_t parsed = (uint64_t)nondet_ulong();
  __ESBMC_assume(parsed <= (uint64_t)LONG_MAX);
  (void)read_mem_spec_gigabytes_unsafe(parsed);
  return 0;
}
