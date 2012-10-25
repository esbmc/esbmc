// queue::front
#include <iostream>
#include <queue>
#include <cassert>
using namespace std;

int main ()
{
  queue<int> myqueue;

  myqueue.push(77);
  myqueue.push(16);
  assert(myqueue.front() == 77);
  myqueue.front() -= myqueue.back();    // 77-16=61
  assert(myqueue.front() != 61);
  cout << "myqueue.front() is now " << myqueue.front() << endl;

  return 0;
}
