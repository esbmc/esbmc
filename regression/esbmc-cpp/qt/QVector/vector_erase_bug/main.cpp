#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> vector;
    QVector<int> ::iterator it;
    vector << 1 << 4 << 6 << 8 << 10 << 12;

    vector.erase(vector.begin());

    assert( !(vector.size() == 5) );

  return 0;
}
