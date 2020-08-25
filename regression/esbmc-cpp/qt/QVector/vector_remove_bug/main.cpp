#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> myQvector;

    myQvector << 1 << 5 << 9;

    myQvector.remove(1);

    assert( !(myQvector.size() == 2) );

  return 0;
}

