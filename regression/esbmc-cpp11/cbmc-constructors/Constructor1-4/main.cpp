#include <cassert>
class t2
{
public:
  int i;

  t2() : i(2)
  {
  }
};

int main()
{
  t2 *p = new t2;
  assert(p->i == 2);
  delete p;
}

#if 0
Symbol......: t2::t2(this)
Pretty name.:
Module......: main
Base name...: t2
Mode........: cpp
Type........: auto (struct t2 *) -> constructor
Value.......: {
  this->i = 2;
}
Flags.......:
Location....: file Constructor1-4/main.cpp line 7
#endif
