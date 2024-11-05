// example.c
#include <assert.h>
struct buffer {	
  int size;
};

typedef struct buffer buffer;

int main () {
  int counter = 6;
  buffer buff;
  buff.size = 6;
  for (int i = 0; i < buff.size; i++) {
    counter--;				
  }	
  assert (counter == 0);
  return counter;
}

