
extern int printf ( const char * format, ... );

int main(void) {
    int *myPointerA = ((void*) 0);
    int *myPointerB = ((void*) 0);

    {
        int myNumberA = 7;
        myPointerA = &myNumberA;
        // scope of myNumber ends here
    }

    int myNumberB = 3;
    myPointerB = &myNumberB;

    int sumOfMyNumbers = *myPointerA + *myPointerB; // myPointerA is out of scope
    printf("%d", sumOfMyNumbers);

    return 0;
}
