// github.com/esbmc/esbmc/issues/4245 — std::chrono shim.
#include <chrono>
#include <cassert>

int main()
{
  std::chrono::microseconds us(100);
  assert(us.count() == 100);

  std::chrono::milliseconds ms(2);
  assert(ms.count() == 2);

  auto sum = us + std::chrono::microseconds(50);
  assert(sum.count() == 150);

  auto diff = std::chrono::milliseconds(5) - std::chrono::milliseconds(2);
  assert(diff.count() == 3);

  assert(us < std::chrono::microseconds(101));
  assert(us == std::chrono::microseconds(100));

  auto as_us = std::chrono::duration_cast<std::chrono::microseconds>(ms);
  assert(as_us.count() == 2000);

  std::chrono::seconds s(1);
  auto s_as_ms = std::chrono::duration_cast<std::chrono::milliseconds>(s);
  assert(s_as_ms.count() == 1000);

  return 0;
}
