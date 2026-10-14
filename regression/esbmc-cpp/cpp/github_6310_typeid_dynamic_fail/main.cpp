// github #6310, negative direction: the dynamic type must be able to make a
// typeid comparison false, so the checks in github_6310_typeid_dynamic are not
// vacuously discharged.
#include <cassert>
#include <typeinfo>

struct A
{
  virtual ~A()
  {
  }
};
struct B : A
{
};
struct C : A
{
};

int main()
{
  A *p = new B();
  assert(typeid(*p) == typeid(C));
  delete p;
  return 0;
}
