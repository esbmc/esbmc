#include <cstdint>

static uint64_t read_mem_spec_gigabytes_unsafe(uint64_t parsed_value)
{
  const uint64_t mult = 1024ULL * 1024ULL * 1024ULL;
  return parsed_value * mult;
}

int main(void)
{
  uint64_t parsed = nondet_ulong();
  (void)read_mem_spec_gigabytes_unsafe(parsed);
  return 0;
}
