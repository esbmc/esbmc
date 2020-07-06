// QQueue::enqueue/dequeue
#include <iostream>
#include <QQueue>
#include <cassert>
using namespace std;

int main ()
{
  QQueue<int> myQQueue;
  int myint;

  cout << "Please enter some integers (enter 0 to end):\n";
   int i;
for(i = 0;i < 5;i++)
    myQQueue.enqueue(i);
   
  cout << "myQQueue contains: ";
  while (!myQQueue.isEmpty())
  {
   assert(myQQueue.head() != 5-i--);//cout << " " << myQQueue.head()<< i-- << endl;
    myQQueue.dequeue();
  }

  return 0;
}
