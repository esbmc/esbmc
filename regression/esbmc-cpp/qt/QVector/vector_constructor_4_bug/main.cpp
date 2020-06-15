#include <cassert>
#include <QVector>

int main ()
{
    QVector <int> myQVector2;
    QVector <int> myQVector(myQVector2);

    assert( !(myQVector.isEmpty()) );

    return 0;
}
