#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> vector;
    //int iRet;
    vector << 1 << 4 << 6 << 8 << 10 << 12;

    assert( !(vector.count() > 0) );

  return 0;
}

