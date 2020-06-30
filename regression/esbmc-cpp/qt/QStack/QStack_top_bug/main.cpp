// QStack::top
#include <iostream>
#include <QStack>
#include <cassert>
using namespace std;

int main ()
{
  QStack<int> myQStack;

  myQStack.push(10);
  myQStack.push(20);

  myQStack.top() -= 5;

  cout << "myQStack.top() is now " << myQStack.top() << endl;
  assert(myQStack.top() != 15);
  return 0;
}
