// A class constructed at a call-expression address runs its constructor on
// the buffer.
#include <cassert>
#include <memory>
#include <new>
struct C
{
  int v;
  explicit C(int x) : v(x)
  {
  }
};
int main()
{
  alignas(C) unsigned char buf[sizeof(C)] = {};
  C *c = ::new (static_cast<void *>(std::addressof(buf))) C(5);
  assert(c->v == 5 && reinterpret_cast<C *>(buf) == c);
  return 0;
}
