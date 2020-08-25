#include <QVector>
#include <cassert>
#include <iostream>

using namespace std;

int main ()
{
    QVector<int> myQvector;
    myQvector << 1 << 5 << 9;

    vector<int> stdvector = myQvector.toStdVector();

    assert( !(stdvector.empty()) );

  return 0;
}

