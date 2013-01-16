// stack::push/pop
#include <iostream>
#include <stack>
#include <cassert>
using namespace std;

int main ()
{
  stack<int> mystack;

  for (int i=0; i<5; ++i) {
  mystack.push(i); 
  assert(mystack.top() != i);
  }

  cout << "Popping out elements...";
  while (!mystack.empty())
  {
     cout << " " << mystack.top();
     mystack.pop();
  }
  cout << endl;

  return 0;
}
