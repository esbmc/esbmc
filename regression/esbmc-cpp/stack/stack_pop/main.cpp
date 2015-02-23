// stack::push/pop
#include <iostream>
#include <stack>
#include <cassert>
using namespace std;

int main ()
{
  stack<int> mystack;

  for (int i=0; i<5; ++i) mystack.push(i);
  int n;
  cout << "Popping out elements...";
  while (!mystack.empty())
  {
     n = mystack.size();
     cout << " " << mystack.top();
     mystack.pop();
     assert(n - 1 == mystack.size());
  }
  assert(mystack.empty());
  cout << endl;

  return 0;
}
