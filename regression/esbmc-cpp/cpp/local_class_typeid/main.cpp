// Each local class is its own type ([class.local]), so same-named local
// classes in different functions or blocks have distinct type_info objects.
#include <typeinfo>
#include <cassert>

struct S
{
};

struct B
{
  virtual ~B() {}
};

const std::type_info &f()
{
  struct S
  {
  };
  return typeid(S);
}

const std::type_info &f(int)
{
  struct S
  {
  };
  return typeid(S);
}

const std::type_info &g(bool k)
{
  if (k)
  {
    struct S
    {
    };
    return typeid(S);
  }
  struct S
  {
  };
  return typeid(S);
}

const std::type_info &h()
{
  struct D : B
  {
  };
  static D d;
  B *p = &d;
  return typeid(*p);
}

const std::type_info &i()
{
  struct D : B
  {
  };
  static D d;
  B *p = &d;
  return typeid(*p);
}

int main()
{
  assert(f() == f());
  assert(f() != typeid(S));
  assert(f() != f(0));
  assert(g(true) != g(false));
  assert(g(true) != f());
  assert(h() != i());
  return 0;
}
