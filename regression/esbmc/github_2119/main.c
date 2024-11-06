// example.c
#include <assert.h>

struct buffer {
  int size;
  int limit;
  int counter[5];
} buff;

typedef struct buffer buffer;

int main () {
  int counter = 6;
  buff.size = 6;
  buff.limit = 3;

  for (int i = 0; i < buff.size; i++) {
    if(buff.limit == i && !buff.counter[i])
      counter--;				
  }
	
  assert (counter == 5);

  return counter;
}

