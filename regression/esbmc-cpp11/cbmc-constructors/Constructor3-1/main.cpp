#include <cassert>
class x
{
public:
  x();
  x(int z);
  int i;
};

x::x() : i(1)
{
}
x::x(int z) : i(z)
{
}

int main()
{
  x a(5); // struct x a;
  assert(a.i == 5); // x(this,signed_int)(&a, 5);
}

#if 0
// class
Symbol......: tag-x
Pretty name.: x
Module......: main
Base name...: x
Mode........: cpp
Type........: class
Value.......:
Flags.......: type
Location....: file Constructor3-1/main.cpp line 2

//x::x()
Symbol......: x::x(this)
Pretty name.:
Module......: main
Base name...: x
Mode........: cpp
Type........: auto (struct x *) -> constructor
Value.......: {
  this->i = 1;
}
Flags.......:
Location....: file Constructor3-1/main.cpp line 5

//x::x(int z);
Symbol......: x::x(this,signed_int)
Pretty name.:
Module......: main
Base name...: x
Mode........: cpp
Type........: auto (struct x *, signed int) -> constructor
Value.......: {
  this->i = z;
}
Flags.......:
Location....: file Constructor3-1/main.cpp line 6
#endif
