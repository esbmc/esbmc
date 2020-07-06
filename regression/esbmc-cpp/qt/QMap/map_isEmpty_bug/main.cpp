#include <cassert>
#include <QMap>

//TODO a função isEmpty não está funcionando de acordo como a documentação

int main ()
{
    QMap<int, int> myQMap;
    QMap<int, int> :: const_iterator it;

    myQMap[1] = 500;
    myQMap[2] = 300;
    myQMap[3] = 100;

    assert( myQMap.isEmpty() );

    return 0;
}

