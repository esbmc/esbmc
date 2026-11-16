#include <cassert>
#include <optional>

struct C
{
  static int made, destroyed;
  int x;
  C() : x(7)
  {
    ++made;
  }
  C(int v) : x(v)
  {
    ++made;
  }
  C(const C &o) : x(o.x)
  {
    ++made;
  }
  ~C()
  {
    ++destroyed;
  }
};
int C::made = 0;
int C::destroyed = 0;

struct NoDefault
{
  int v;
  NoDefault(int x) : v(x)
  {
  }
};

int main()
{
  {
    std::optional<C> a;
    std::optional<C> b(std::nullopt);
    assert(C::made == 0 && !a && !b);
  }
  assert(C::destroyed == 0);

  {
    std::optional<C> o;
    o.emplace(5);
    assert(C::made == 1 && o->x == 5);
    std::optional<C> p = o;
    assert(C::made == 2 && p->x == 5);
    o.reset();
    assert(C::destroyed == 1 && !o);
    o = p;
    assert(o->x == 5);
    o = std::nullopt;
    assert(!o);
  }
  assert(C::destroyed == C::made);

  std::optional<NoDefault> nd;
  assert(!nd);
  nd = NoDefault(3);
  assert(nd->v == 3);

  std::optional<int> i, j(2);
  i.swap(j);
  assert(*i == 2 && !j);
  return 0;
}
