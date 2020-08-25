#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> vector;
    vector << 1 << 4 << 6;

    vector.resize(6);

    assert( vector.size() == 6 );

  return 0;
}
