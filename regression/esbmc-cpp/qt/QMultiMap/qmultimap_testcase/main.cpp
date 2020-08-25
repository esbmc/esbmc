#include <QMultiMap>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    
    QMultiMap<QString, int> map1, map2, map3;
    
    map1.insert("plenty", 100);
    map1.insert("plenty", 2000);
    assert(map1.size() == 2);
    
    map2.insert("plenty", 5000);
    assert(map2.size() == 1);
    
    //map3 = map1 + map2;
    //assert(map3.size() == 3);
    
    return 0;
}