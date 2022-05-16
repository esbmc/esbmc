#include <cassert>
class t3
{
public:
  int i;

  t3();
};

t3::t3() : i(3)
{
}

int main()
{
  t3 instance3; // struct t3 instance3;
  assert(instance3.i == 3); // t3(this)(&instance3);
}

#if 0
Symbol......: t3::t3(this)
Pretty name.:
Module......: main
Base name...: t3
Mode........: cpp
Type........: auto (struct t3 *) -> constructor
Value.......: {
  this->i = 3;
}
Flags.......:
Location....: file Constructor1-3/main.cpp line 7
#endif
