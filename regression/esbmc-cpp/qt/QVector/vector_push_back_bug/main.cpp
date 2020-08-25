// QVector::push_back
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;
  int myint;

  cout << "Please enter some integers (enter 0 to end):\n";

  do {
    cin >> myint;
    myQVector.push_back (myint);
    assert(myQVector.back() != myint);
  } while (myint);

  cout << "myQVector stores " << (int) myQVector.size() << " numbers.\n";

  return 0;
}
