#include <cassert>
#include <QMap>


int main ()
{
    QMap<int, int> myQMap;
    QMap<int, int> :: const_iterator it;
    int iRet;
    
    myQMap[1] = 500;
    myQMap[2] = 300;
    myQMap[3] = 100;
    
    iRet = myQMap.key(200,-1);
    
    assert(iRet == -1);
    
    return 0;
}
