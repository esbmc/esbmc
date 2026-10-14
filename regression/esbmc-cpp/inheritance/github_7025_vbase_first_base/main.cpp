// Control for github_7025_vbase_nonfirst_member: the same hierarchy with the
// virtual base at offset 0 (B declared first). This shape already resolves
// correctly, so it pins that the working case stays working.
#include <cassert>

struct V
{
  int v;
};
struct A
{
  int a;
};
struct B : virtual V
{
  char buf[8];
  void put(char c)
  {
    buf[0] = c;
  }
};
struct D : B, A
{
};

int main()
{
  D d;
  d.buf[0] = 'x';
  d.put('A');
  assert(d.buf[0] == 'A');
  return 0;
}
