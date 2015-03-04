// priority_queue::top
#include <iostream>
#include <queue>
#include <cassert>
using namespace std;

int main ()
{
  priority_queue<int> mypq;

  mypq.push(10);
  mypq.push(20);
  mypq.push(15);
  assert(mypq.top() == 20);
  cout << "mypq.top() is now " << mypq.top() << endl;

  return 0;
}
