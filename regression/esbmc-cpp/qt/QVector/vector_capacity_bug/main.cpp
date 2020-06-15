#include <QVector>
#include <cassert>

int main ()
{

  QVector<int> myQVector;
  myQVector << 1 << 4 << 6 << 8 << 10 << 12;
  assert( !(myQVector.capacity() > 0) );

  return 0;
}

