#include <cassert>
#include <expected>

struct Err
{
  int code;
};

std::expected<int, Err> compute(bool ok)
{
  if (ok)
    return 100;
  return std::unexpected<Err>(Err{-1});
}

int main()
{
  auto a = compute(true);
  assert(a.has_value());
  assert(*a == 100);
  assert(a.value() == 100);

  auto e = compute(false);
  assert(!e.has_value());
  assert(e.error().code == -1);

  // value_or fallback.
  std::expected<int, int> bad = std::unexpected<int>(7);
  assert(bad.value_or(42) == 42);
  std::expected<int, int> good = 5;
  assert(good.value_or(42) == 5);

  // operator-> on success path.
  std::expected<Err, int> rec = Err{99};
  assert(rec->code == 99);

  return 0;
}
