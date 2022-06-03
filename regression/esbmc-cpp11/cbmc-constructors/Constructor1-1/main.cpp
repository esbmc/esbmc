#include <cassert>
class t1
{
public:
  int i;

  t1()
  {
    i = 1;
  }
};

int main()
{
  t1 instance1; // struct t1 instance1;
  assert(instance1.i == 1); // t1(this)(&instance1);
}

#if 0
// user defined constructor
Symbol......: t1::t1(this)
Pretty name.:
Module......: main
Base name...: t1
Mode........: cpp
Type........: auto (struct t1 *) -> constructor
Value.......: {
  this->i = 1;
}
Flags.......:
Location....: file Constructor1-1/main.cpp line 7
#endif
