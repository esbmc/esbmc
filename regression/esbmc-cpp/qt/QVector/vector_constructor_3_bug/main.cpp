#include <cassert>
#include <QVector>

int main ()
{
    QVector <int> myQVector(3,2);

    assert( !(myQVector.contains(2)) );

    return 0;
}

