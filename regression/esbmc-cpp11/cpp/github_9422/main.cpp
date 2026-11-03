#include <cassert>
#include <memory>
#include <utility>

static int calls = 0;

struct D
{
  void operator()(int *p) const
  {
    ++calls;
    delete p;
  }
};

int main()
{
  {
    std::unique_ptr<int, D> p;
  }
  assert(calls == 0);

  std::unique_ptr<int, D> a(new int(1));
  std::unique_ptr<int, D> b(std::move(a));
  a.reset();
  assert(calls == 0);

  std::unique_ptr<int, D> c;
  c = std::move(a);
  assert(calls == 0);

  b.reset(new int(2));
  assert(calls == 1 && *b == 2);

  c = std::move(b);
  assert(calls == 1 && *c == 2 && !b);
  return 0;
}
