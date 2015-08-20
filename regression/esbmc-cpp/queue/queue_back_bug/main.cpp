// queue::front
#include <iostream>
#include <queue>
#include <cassert>
using namespace std;

int main ()
{
  queue<int> myqueue;

  myqueue.push(12);
  myqueue.push(75);   // this is now the back
  assert(myqueue.back() != 75);
  myqueue.back() -= myqueue.front();
  assert(myqueue.back() != 63);
  cout << "myqueue.back() is now " << myqueue.back() << endl;

  return 0;
}
