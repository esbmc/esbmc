// stack::top
#include <stack>
#include <cassert>

using namespace std;

int main ()
{
  stack<int> mystack;

  mystack.push(10);
  mystack.push(20);

  assert(mystack.top()==20);

  return 0;
}

