// QVector::back
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;
    assert(myQVector.empty());
    myQVector.push_back(10);
    myQVector.push_back(10);
    assert(!myQVector.empty());
  return 0;
}
