#include <cassert>
#include <QHash>

int main ()
{
    QHash<int, int> myQHash1;
    QHash<int, int> myQHash2;
    QHash<int, int> :: const_iterator it;

    myQHash1[1] = 500;
    myQHash1[2] = 300;
    myQHash1[3] = 100;

    myQHash2[1] = 400;
    myQHash2[2] = 900;
    myQHash2[3] = 700;

    myQHash1.unite(myQHash2);

    assert(myQHash1.size() != 6);

    return 0;
}

