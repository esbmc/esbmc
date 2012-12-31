// deque::push_back
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  deque<int> mydeque;
  int myint;

/*  cout << "Please enter some integers (enter 0 to end):\n"; */
  int n = 10;

  do {
    myint = --n;
    mydeque.push_back (myint);
  } while (myint);
  assert(mydeque.back() == 0);
  assert(mydeque.size() == 10);
  cout << "mydeque stores " << (int) mydeque.size() << " numbers.\n";

  return 0;
}
