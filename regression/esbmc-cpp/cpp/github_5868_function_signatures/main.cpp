#include <functional>
#include <cassert>

static int g_seen = 0;
static void sink(int x)
{
  g_seen = x;
}

struct pt
{
  int x;
  int y;
};

int main()
{
  // The previous model routed every call through int invoke(int) /
  // int invoke(int, int), so anything else truncated or did not compile.
  std::function<double(double)> d = [](double x) { return x + 0.5; };
  assert(d(2.0) == 2.5);

  std::function<void(int)> v = sink;
  v(7);
  assert(g_seen == 7);

  std::function<bool(int, int)> b = [](int a, int c) { return a < c; };
  assert(b(1, 2));
  assert(!b(2, 1));

  std::function<int()> n = []() { return 42; };
  assert(n() == 42);

  std::function<int(pt)> s = [](pt p) { return p.x + p.y; };
  pt p;
  p.x = 3;
  p.y = 4;
  assert(s(p) == 7);

  std::function<int(int, int, int)> t = [](int a, int c, int e) {
    return a + c + e;
  };
  assert(t(1, 2, 3) == 6);

  assert(static_cast<bool>(d));
  std::function<int(int)> empty;
  assert(!static_cast<bool>(empty));
  return 0;
}
