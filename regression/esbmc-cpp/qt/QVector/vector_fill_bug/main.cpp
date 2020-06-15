#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> vector(5);

    vector.fill(99);

    assert( !(vector.contains(99)) );

  return 0;
}
