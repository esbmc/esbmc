#include <assert.h>

struct temp{
    long x;
    long y;
};

void loop1(){
    for(int i=0;i<1;i++){
        assert(1);
    }
}


void loop2(){
    for(int i=0;i<1;i++){
        assert(1);
    }
}

void fun(unsigned long x,unsigned long y){

    assert(x == 1);

    switch(x){
    case 1:
        loop1();
        return;
    case 2:
        loop2();
        return;
    default:
        return;
    }
}

int main(){
    struct temp temp1;
    temp1.x = 1;
    temp1.y = 2;
    struct temp temp2 = {1};
    fun(temp1.x, temp1.y);
    return 0;
}

