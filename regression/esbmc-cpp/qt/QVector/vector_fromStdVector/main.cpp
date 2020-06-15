#include <QVector>
#include <cassert>
#include <iostream>

using namespace std;

int main ()
{
    std::vector<int> stdvector;
    stdvector.push_back(1);
    stdvector.push_back(5);
    stdvector.push_back(3);

    QVector<int> vector = QVector<int>::fromStdVector(stdvector);

    assert( !(vector.isEmpty()) );

  return 0;
}
