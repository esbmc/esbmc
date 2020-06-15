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

    myQHash2[1] = 500;
    myQHash2[2] = 300;
    myQHash2[3] = 100;

    assert(myQHash1 == myQHash2);


    return 0;
}
