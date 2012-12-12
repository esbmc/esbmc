// erasing from deque
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  unsigned int i;
  deque<unsigned int> mydeque;

  // set some values (from 1 to 10)
  for (i=1; i<=10; i++) mydeque.push_back(i);
  
  // erase the 6th element
  mydeque.erase (mydeque.begin()+5);
  assert(mydeque[5] == 7);

  // erase the first 3 elements:
  mydeque.erase (mydeque.begin(),mydeque.begin()+3);
  assert(mydeque.front() == 4);

  cout << "mydeque contains:";
  for (i=0; i<mydeque.size(); i++)
    cout << " " << mydeque[i];
  cout << endl;

  return 0;
}
