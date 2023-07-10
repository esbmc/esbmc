// TC description:
//  nested class using its type in its method. The type is incomplete.
#include <cassert>

char x = 'X';

class string
{
public:
  class iterator
  {
  public:
    char* pointer;
    void do_something()
    {
      // when using its type in side its function, the type is incomplete.
      iterator buffer;
      buffer.pointer = &x;
      pointer = buffer.pointer;
    }
  };

  iterator itr;
};

int main()
{
  string myString;
  myString.itr.do_something();
  assert(myString.itr.pointer == &x);
  assert(myString.itr.pointer != &x); // should be identical
  return 0;
}
