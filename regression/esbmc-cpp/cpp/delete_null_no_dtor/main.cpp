#include <cassert>
int dtor_count;
struct A
{
  virtual ~A() { dtor_count++; }
};
int main()
{
  {
    A *p = nullptr;
    delete p; // [expr.delete]/7: no destructor call, no-op
  }
  assert(dtor_count == 0);
  return 0;
}
