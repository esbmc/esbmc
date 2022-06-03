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
  t2 instance2; // struct t2 instance2;
  assert(instance2.i == 2); // t2(this)(&instance2);
}

#if 0
// user-defined constructor with initializer list
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
Location....: file Constructor1-2/main.cpp line 8
#endif
