// deque::pop_front
#include <iostream>
#include <deque>
#include <cassert>

using namespace std;

int main ()
{
  deque<int> mydeque;
  int sum (0);
  mydeque.push_back (100);
  mydeque.push_back (200);
  mydeque.push_back (300);

  cout << "Popping out the elements in mydeque:";
//  while (!mydeque.empty())
//  {
//    cout << " " << mydeque.front();
    assert(mydeque.front()==100);
    mydeque.pop_front();
//  }

  cout << "\nFinal size of mydeque is " << int(mydeque.size()) << endl;
  assert(int(mydeque.size())==0);

  return 0;
}

