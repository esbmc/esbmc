#include <cassert>
#include <QHash>

int main ()
{
    QHash<int, int> myQHash;
    QHash<int, int>::iterator it;

    myQHash[1] = 500;
    myQHash[2] = 300;
    myQHash[3] = 100;

    it = myQHash.begin();

    it = myQHash.find(1);
    if (it.value() == 500) {
        it = myQHash.erase(it);
    }
    
    it = myQHash.find(2);
    if (it.value() == 300) {
        it = myQHash.erase(it);
    }

    it = myQHash.find(3);
    if (it.value() == 100) {
        it = myQHash.erase(it);
    }

    assert(myQHash.empty());

    return 0;
}
