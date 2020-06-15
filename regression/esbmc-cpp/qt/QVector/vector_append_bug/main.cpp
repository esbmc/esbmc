#include <QVector>
#include <cassert>

int main ()
{
  QVector<int> myQVector;

  myQVector.append(10);
  int n = 10;
  while (myQVector.back() != 0)
  {
    assert(myQVector.back() != n--);
    myQVector.push_back ( myQVector.back() -1 );
  }

  return 0;
}
