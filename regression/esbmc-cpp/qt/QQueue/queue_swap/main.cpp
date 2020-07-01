// constructing QQueues
#include <iostream>
#include <QQueue>
#include <cassert>
using namespace std;

int main ()
{
    QQueue<int> x;
	QQueue<int> others;
    x.enqueue(1);
    x.enqueue(2);
    x.enqueue(3);
	x.swap(others);
    assert(others.size() == 3);
	others.swap(x);
    while (!x.isEmpty())
        cout << x.dequeue() << endl;
    assert(x.isEmpty());
  return 0;
}
