// queue::push/pop
#include <iostream>
#include <queue>
#include <cassert>
using namespace std;

int main ()
{
  queue<int> myqueue;
  int myint;

  cout << "Please enter some integers (enter 0 to end):\n";
   int i;
for(i = 0;i < 5;i++)
    myqueue.push(i);
   
  cout << "myqueue contains: ";
  while (!myqueue.empty())
  {
   assert(myqueue.front() == 5-i--);//cout << " " << myqueue.front()<< i-- << endl;
    myqueue.pop();
  }

  return 0;
}
