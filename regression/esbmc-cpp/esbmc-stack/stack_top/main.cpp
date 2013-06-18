// stack::top
#include <iostream>
#include <stack>
#include <cassert>
using namespace std;

int main ()
{
  stack<int> mystack;

  mystack.push(10);
  mystack.push(20);

  mystack.top() -= 5;

  cout << "mystack.top() is now " << mystack.top() << endl;
  assert(mystack.top() == 15);
  return 0;
}
