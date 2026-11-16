#include <cassert>
#include <memory>

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
  assert(calls != 0);
  return 0;
}
