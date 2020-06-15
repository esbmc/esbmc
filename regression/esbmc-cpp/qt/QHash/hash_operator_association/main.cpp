#include <cassert>
#include <QHash>

int main ()
{
    QHash<int, int> myQHash1;

    myQHash1[1] = 500;
    myQHash1[2] = 300;
    myQHash1[3] = 100;

    assert(myQHash1.size() != 0);

    return 0;
}
