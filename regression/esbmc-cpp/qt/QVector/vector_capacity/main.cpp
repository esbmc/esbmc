#include <QVector>
#include <cassert>

int main ()
{

  QVector<int> myQVector(5,2);
  assert( myQVector.capacity() == 5 );

  return 0;
}

