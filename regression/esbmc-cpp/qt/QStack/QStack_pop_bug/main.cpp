// QStack::push/pop
#include <iostream>
#include <QStack>
#include <cassert>
using namespace std;

int main ()
{
  QStack<int> myQStack;

  for (int i=0; i<5; ++i) myQStack.push(i);
  int n;
  cout << "Popping out elements...";
  while (!myQStack.isEmpty())
  {
     n = myQStack.size();
     cout << " " << myQStack.top();
     myQStack.pop();
     assert(n - 1 != myQStack.size());
  }
  assert(myQStack.isEmpty());
  cout << endl;

  return 0;
}
