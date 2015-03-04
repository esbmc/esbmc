// deque::front
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  deque<int> mydeque;

  mydeque.push_front(77);
  assert(mydeque.front() != 77);
  mydeque.push_back(16);

  mydeque.front() -= mydeque.back();
  assert(mydeque.front() != 61);
  cout << "mydeque.front() is now " << mydeque.front() << endl;

  return 0;
}
