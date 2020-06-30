#include <QVector>
#include <QList>
#include <cassert>

int main ()
{
    QVector<int> vect;
    vect << 1 << 6 << 8 << 12;

    QList<int> list = vect.toList();
    // list: [1, 6, 8, 12]

    assert( !(list.isEmpty()) );

  return 0;
}

