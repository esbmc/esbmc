// QQueue::head
#include <iostream>
#include <QQueue>
#include <cassert>
using namespace std;

int main ()
{
  QQueue<int> myQQueue;

  myQQueue.enqueue(77);
  myQQueue.enqueue(16);
  assert(myQQueue.head() == 77);
  myQQueue.head() -= myQQueue.back();    // 77-16=61
  assert(myQQueue.head() != 61);
  cout << "myQQueue.head() is now " << myQQueue.head() << endl;

  return 0;
}
