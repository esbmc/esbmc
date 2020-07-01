#include <cassert>
#include <QMap>

//TODO a função insertMulti nao está funcionando conforme a documentação

int main ()
{
    QMap<int, int> myQMap;
    QMap<int, int> :: const_iterator it;

    myQMap[1] = 500;
    myQMap[2] = 300;
    myQMap[3] = 100;

    it = myQMap.insertMulti(4,900);

    assert( myQMap.contains(it.key()) );

    return 0;
}

