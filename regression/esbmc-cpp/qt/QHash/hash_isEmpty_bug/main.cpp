#include <cassert>
#include <QHash>
#include<iostream>

using namespace std;

int main ()
{
    QHash<int, int> myQHash;
    QHash<int, int> :: const_iterator it;
    bool bRet;

    myQHash[1] = 500;
    myQHash[2] = 300;
    myQHash[3] = 100;

    bRet = myQHash.isEmpty();

    assert(bRet == true);

    return 0;
}

