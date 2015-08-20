// clearing deques
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  unsigned int i;
  deque<int> mydeque;
  mydeque.push_back (100);
  mydeque.push_back (200);
  mydeque.push_back (300);

  cout << "mydeque contains:";
  for (i=0; i<mydeque.size(); i++) cout << " " << mydeque[i];

  mydeque.clear();
  assert(!mydeque.empty());
  mydeque.push_back (1101);
  mydeque.push_back (2202);
  assert(mydeque.size() != 2);
  cout << "\nmydeque contains:";
  for (i=0; i<mydeque.size(); i++) cout << " " << mydeque[i];

  cout << endl;

  return 0;
}
