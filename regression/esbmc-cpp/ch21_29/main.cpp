// queue::empty
#include <iostream>
#include <queue>
#include <cassert>

using namespace std;

int main ()
{
  queue<int> myqueue;
  int sum (0);

  for (int i=1;i<=10;i++) myqueue.push(i);

  while (!myqueue.empty())
  {
     sum += myqueue.front();
     myqueue.pop();
  }

  cout << "total: " << sum << endl;
  assert(sum==55);

  return 0;
}
