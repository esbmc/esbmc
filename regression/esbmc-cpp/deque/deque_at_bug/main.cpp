// deque::at
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  deque<int> mydeque (10);   // 10 zero-initialized ints
  unsigned int i;

  // assign some values:
  for (i=0; i<mydeque.size(); i++)
    mydeque.at(i)=i;

  cout << "mydeque contains:";
  for (i=0; i<mydeque.size(); i++)
    assert(mydeque.at(i) != i);

  cout << endl;

  return 0;
}
