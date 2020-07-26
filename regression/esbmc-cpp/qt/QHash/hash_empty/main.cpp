#include <cassert>
#include <QHash>
#include<iostream>

using namespace std;

int main ()
{
    QHash<int, int> myQHash;
    QHash<int, int> :: const_iterator it;
    bool bRet;

    bRet = myQHash.empty();

    assert(bRet == true);

    return 0;
}
