#include <cassert>
#include <QVector>

int main ()
{
    QVector <int> myQVector(3);

    assert( !(myQVector.size() == 3) );


    return 0;
}

