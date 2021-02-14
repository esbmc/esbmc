typedef long unsigned int size_t;
extern int printf (const char* format, ...);
extern void* alloca(size_t size);

// function returns array of numbers
int * getNumbers(void) {

   int *array = alloca(10 * sizeof(int)); // array should be static

   for (int i = 0; i < 10; ++i) {
      array[i] = i;
   }

   return array;
}

int main(void) {

   int *numbers = getNumbers();

   for (int i = 0; i < 10; i++ ) {
      printf( "%d\n", *(numbers + i));
   }

   return 0;
}
