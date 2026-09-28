#include <cassert>
int dtor_count;
struct A
{
  virtual ~A() { dtor_count++; }
};
int main()
{
  A *p = new A();
  delete p; // destructor must still run for non-null
  assert(dtor_count == 0); // must fail: non-null delete runs the destructor
  return 0;
}
