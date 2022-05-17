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

class t2
{
public:
  int i;

  t2() : i(2)
  {
  }
};

class t3
{
public:
  int i;

  t3(int x);

  int get_i()
  {
    return i;
  }
};

t3::t3(int x)
{
  i = x;
}

int main()
{
  t1 instance1;
  assert(instance1.i == 1); // public memebr access

  t2 instance2;
  assert(instance2.i == 2); // different class constructor

  t3 instance3(3);
  assert(instance3.i == 3);

  t3 instance4(4);
  assert(instance4.i == 4); // another instantiation of t3 with different ctor param

  t3 instance5(5);
  assert(instance5.get_i() == 5); // member function access

  t3 instance6(6);
  assert(instance6.get_i() == 6); // member function access in another object

  /*
  t2 *p = new t2;
  assert(p->i == 2);
  delete p;
  */
}

#if 0
#endif
