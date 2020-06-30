#include <QVector>
#include <cassert>

int main ()
{
    QVector<double> vector;
    vector << 2.718 << 1.442 << 0.4342;
    vector.insert(1, 3);
    // vector: [2.718, 3, 1.442, 0.4342]

    assert( vector.size() == 4 );

  return 0;
}

