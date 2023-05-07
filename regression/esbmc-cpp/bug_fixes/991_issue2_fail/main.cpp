#include <cassert>

char x = 'X';

class string
{
public:
  string() {}
  class iterator
  {
  public:
    iterator() { data = 1; pointer = &x; }
    char* pointer;
    int data;
  };

  iterator itr;
};

int main()
{
  string myString;
  assert(myString.itr.data == 1);
  assert(myString.itr.pointer != &x); // FAIL, should be the same
  return 0;
}
