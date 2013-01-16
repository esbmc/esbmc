// priority_queue::push/pop
#include <iostream>
#include <queue>
#include <cassert>
using namespace std;

int main ()
{
  priority_queue<int> mypq;
  int arrae[4] = {100, 40, 30, 25};
  int i = 0;
  mypq.push(30);
  mypq.push(100);
  mypq.push(25);
  mypq.push(40);

  cout << "Popping out elements...";
  while (!mypq.empty())
  {
     cout << " " << mypq.top();
     assert(mypq.top() == arrae[i++]);
     mypq.pop();
  }
  cout << endl;

  return 0;
}
