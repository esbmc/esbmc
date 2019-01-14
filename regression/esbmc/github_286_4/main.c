
extern int printf (const char* format, ...);

// function returns array of numbers
int* getNumbers() {

    static int array[10];

    for (int i = 0; i < 10; ++i) {
       array[i] = i;
    }

    return array;
}

int* getNumbers2() {
    int* numbers = getNumbers();
    static int numbers2[10];
    for (int i = 0; i < 10; ++i) {
        numbers2[i] = numbers[i];
    }
    return numbers2;
}

int* getNumbers3() {
    int* numbers = getNumbers2();
    int numbers3[10];
    for (int i = 0; i < 10; ++i) {
        numbers3[i] = numbers[i];
    }

    return numbers3;
}

int* getNumbers4() {
    int* numbers = getNumbers3();
    static int numbers4[10];
    for (int i = 0; i < 10; ++i) {
        numbers4[i] = numbers[i];
    }
    return numbers4;
}

int main (void) {

    int *numbers = getNumbers4();

    for (int i = 0; i < 10; i++ ) {
       printf( "%d\n", *(numbers + i));
    }

    return 0;
}
