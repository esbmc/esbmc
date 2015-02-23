// stack::top
#include <QStack>
#include <cassert>

int main ()
{
  QStack<int> mystack;

  mystack.push(10);
  mystack.push(20);

  assert(mystack.top()==20);

  return 0;
}
