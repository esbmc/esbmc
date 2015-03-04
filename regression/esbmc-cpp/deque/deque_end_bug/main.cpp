//TEST FAILS
// deque::end
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  deque<int> mydeque1;
  deque<int> mydeque2;
  deque<int>::iterator it;

  for (int i=1; i<=5; i++) mydeque1.insert(mydeque1.end(),i);
  for (int i=1; i<=5; i++) mydeque2.push_back(i);
  
  assert(mydeque1 != mydeque2);
  assert(*(mydeque1.end()) != 0);
  cout << "mydeque contains:";

  it = mydeque1.begin();
  assert(*it == 1);
  while (it != mydeque1.end() )
    cout << " " << *it++;

  cout << endl;

  return 0;
}
