#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string_view>

static uint64_t read_time_spec(std::string_view str)
{
  if (str.empty())
    return 0;
  uint64_t mult = 1;
  if (!isdigit((unsigned char)str.back()))
  {
    switch (str.back())
    {
    case 's': mult = 1; break;
    case 'm': mult = 60; break;
    case 'h': mult = 3600; break;
    case 'd': mult = 86400; break;
    default: return 0;
    }
  }
  assert(mult != 0);
  uint64_t timeout = (uint64_t)strtol(str.data(), nullptr, 10);
  assert(timeout <= UINT64_MAX / mult);
  timeout *= mult;
  return timeout;
}

static uint64_t read_mem_spec(std::string_view str)
{
  if (str.empty())
    return 0;
  uint64_t mult = 1024ULL * 1024ULL;
  if (!isdigit((unsigned char)str.back()))
  {
    switch (str.back())
    {
    case 'b': mult = 1; break;
    case 'k': mult = 1024; break;
    case 'm': mult = 1024ULL * 1024ULL; break;
    case 'g': mult = 1024ULL * 1024ULL * 1024ULL; break;
    default: return 0;
    }
  }
  assert(mult != 0);
  uint64_t size = (uint64_t)strtol(str.data(), nullptr, 10);
  assert(size <= UINT64_MAX / mult);
  size *= mult;
  return size;
}

int main(void)
{
  char buf[3];
  unsigned int digit_idx = nondet_uint();
  __ESBMC_assume(digit_idx <= 9);
  buf[0] = (char)('0' + digit_idx);
  buf[1] = nondet_char();
  buf[2] = '\0';

  std::string_view sv2(buf, 2);
  (void)read_time_spec(sv2);
  (void)read_mem_spec(sv2);

  std::string_view empty_sv("", 0);
  assert(read_time_spec(empty_sv) == 0);
  assert(read_mem_spec(empty_sv) == 0);

  char single[2] = {'5', '\0'};
  std::string_view sv1(single, 1);
  assert(read_time_spec(sv1) == 5);
  assert(read_mem_spec(sv1) == 5ULL * 1024ULL * 1024ULL);

  return 0;
}
