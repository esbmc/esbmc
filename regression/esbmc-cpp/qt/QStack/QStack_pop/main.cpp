// QStack::push/pop
#include <iostream>
#include <QStack>
#include <cassert>
using namespace std;

int main ()
{
  QStack<int> myQStack;
  for (int i=0; i<5; ++i) myQStack.push(i);
  cout << " " << myQStack.top();
  myQStack.pop();
  assert(myQStack.size() == 4);
  cout << endl;

  return 0;
}
