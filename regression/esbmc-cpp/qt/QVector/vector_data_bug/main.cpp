#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> vector(5);
    int *data = vector.data();
    for (int i = 0; i < 5; ++i)
    {
        data[i] = 2 * i;
    }

    assert( data == NULL );

  return 0;
}

