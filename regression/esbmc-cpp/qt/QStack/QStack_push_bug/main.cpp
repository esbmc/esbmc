// QStack::push/pop
#include <iostream>
#include <QStack>
#include <cassert>
using namespace std;

int main ()
{
  QStack<int> myQStack;

  for (int i=0; i<5; ++i) {
  myQStack.push(i); 
  assert(myQStack.top() != i);
  }

  cout << "Popping out elements...";
  while (!myQStack.isEmpty())
  {
     cout << " " << myQStack.top();
     myQStack.pop();
  }
  cout << endl;

  return 0;
}
