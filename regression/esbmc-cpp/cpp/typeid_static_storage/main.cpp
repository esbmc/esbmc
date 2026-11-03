// typeid refers to an object of static storage duration ([expr.typeid]/1), so
// a pointer to it, or a type_index built from it, outlives the expression.
#include <typeindex>
#include <typeinfo>
#include <cassert>

struct B
{
  virtual ~B()
  {
  }
};

struct D : B
{
};

int main()
{
  const std::type_info *p = &typeid(int);
  std::type_index i(typeid(int));
  assert(*p == typeid(int));
  assert(i == std::type_index(typeid(int)));
  assert(*p != typeid(char));

  D d;
  B *b = &d;
  std::type_index k(typeid(*b));
  assert(k == std::type_index(typeid(D)) && k != std::type_index(typeid(B)));
  return 0;
}
