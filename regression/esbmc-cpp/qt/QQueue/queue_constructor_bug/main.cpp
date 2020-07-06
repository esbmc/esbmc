// constructing QQueues
#include <iostream>
#include <QQueue>
#include <cassert>
using namespace std;

int main ()
{
    QQueue<int> x;
    x.enqueue(1);
    x.enqueue(2);
    x.enqueue(3);
    while (!x.isEmpty())
        cout << x.dequeue() << endl;
    assert(!x.isEmpty());
  return 0;
}
