#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> myQvector;

    myQvector << 1 << 5 << 9 << 4 << 6 << 7;

    myQvector.remove(1,3);

    assert( myQvector.size() == 3 );

  return 0;
}

