#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> vector;
    vector << 1 << 4 << 6 << 8 << 10 << 12;
    vector.erase(vector.begin(),vector.end());
   // vector: []

    assert(vector.size() == 0);

  return 0;
}

