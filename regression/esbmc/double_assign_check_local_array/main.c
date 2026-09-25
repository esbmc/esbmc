// Copy of regression/esbmc/github_286_3/main.c, run with --double-assign-check.
// Pins R14: the write through the returned dangling pointer must take a fresh
// L2 index rather than re-issue ...@F@getNumbers2@numbers2?1!0&0#1, which the
// declaration of numbers2 already defined.

int array[10];

// function returns array of numbers
int* getNumbers(void) {
    for (int i = 0; i < 10; ++i) {
       array[i] = i;
    }

    return array;
}

int* getNumbers2(void) {
    int* numbers = getNumbers();
    // numbers2 is local
    int numbers2[10];

    for (int i = 0; i < 10; ++i) {
        numbers2[i] = numbers[i];
    }

    return numbers2;
}

int main(void) {
   int *numbers = getNumbers2();
   numbers[0] = 100;

   return 0;
}
