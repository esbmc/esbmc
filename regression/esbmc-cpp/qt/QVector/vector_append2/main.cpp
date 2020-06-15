#include <QVector>
#include <cassert>

//TODO esse benchmark possui um erro ele não adiciona todo os elementos do vector no final do outro conforme explica a documentação

int main ()
{
    QVector<int> vector1;
    QVector<int> vector2;
    vector1 << 1 << 4 << 6;
    vector2 << 6 << 1 << 4;

    vector1.append(vector2);

    assert( vector1.size() == 3 );

  return 0;
}

