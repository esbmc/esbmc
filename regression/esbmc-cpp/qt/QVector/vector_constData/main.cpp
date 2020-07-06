#include <QVector>
#include <cassert>

int main ()
{
    int myints[] = {75,23,65,42,13};
    QVector<int> myQVector;
    QVector<int>::const_iterator it;

    for(int i = 0; i < 5; i++)
    {
        myQVector.push_back(myints[i]);
    }

    assert( myQVector.constData()[0] == 75 );

  return 0;
}

