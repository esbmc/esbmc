#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> vector(5);
    const int *data = vector.data();

    assert( !(data == NULL) );

  return 0;
}
