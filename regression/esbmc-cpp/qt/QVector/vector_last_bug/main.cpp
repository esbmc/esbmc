// QVector::back
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;

  myQVector.push_back(10);
  int n = 10;
  while (myQVector.last() != 0)
  {
    assert(myQVector.last() != n--);
    myQVector.push_back ( myQVector.last() -1 );
  }

  return 0;
}
