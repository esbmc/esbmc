// queue::push/pop
#include <iostream>
#include <queue>
using namespace std;

int main ()
{
  queue<int> myqueue;
  int myint;

  cout << "Please enter some integers (enter 0 to end):\n";

  do {
    cin >> myint;
    myqueue.push (myint);
  } while (myint);

  cout << "myqueue contains: ";
  while (!myqueue.empty())
  {
    cout << " " << myqueue.front();
    myqueue.pop();
  }

  return 0;
}

