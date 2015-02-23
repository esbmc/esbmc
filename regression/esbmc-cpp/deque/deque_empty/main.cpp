// deque::empty
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  deque<int> mydeque;
  int sum (0);

  for (int i=1;i<=10;i++) mydeque.push_back(i);

  while (!mydeque.empty())
  {
     sum += mydeque.front();
     mydeque.pop_front();
  }
  assert(mydeque.empty());
  assert(mydeque.size() == 0);
  cout << "total: " << sum << endl;
  
  return 0;
}
