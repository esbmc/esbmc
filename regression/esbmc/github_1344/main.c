#include <assert.h>

struct temp{
 int x;
};
  
int fun(){
  struct temp temp1;
  temp1.x = 1;
  
for(int i = 0; i < 20; i++){
    for(int j = 0; j < 10; j++){
      if(temp1.x)
       assert(1);
    }
  }
}

int main() {
  for(int k = 0; k < 100; k++){
    fun();
  }
  return 0;
}
