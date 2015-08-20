// deque::back
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  deque<int> mydeque;

  mydeque.push_back(10);
  int n = 10;
  while (mydeque.back() != 0)
  {
    assert(mydeque.back() != n--);
    mydeque.push_back ( mydeque.back() -1 );
  }

  cout << "mydeque contains:";
  for (unsigned i=0; i<mydeque.size() ; i++)
    cout << " " << mydeque[i];

  cout << endl;

  return 0;
}
