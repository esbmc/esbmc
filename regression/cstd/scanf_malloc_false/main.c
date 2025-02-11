#include<stdio.h>
#include<stdlib.h>

void search(int arr[], int n, int x) {
    int flag = 0, index = 0;
    for(int i = 0; i < n; i++) {
        if(arr[i] == x) {
            flag = 1;
            index = i;
            break;
        }
    }
    if(flag) {
        printf("%d is found at index %d", x, index);
    } else {
        printf("%d is not present in array", x);
    }
}

int main() {
    int n, x;
    printf("Enter size of array: ");
    scanf("%d", &n);

    int *arr = (int*) malloc(10 * sizeof(int));
    printf("Enter %d elements: ", n);
    for(int i = 0; i < 10; i++) {
        scanf("%d", &arr[i]);
    }

    printf("Enter element to search: ");
    scanf("%d", &x);

    search(arr, n, x);

    return 0;
}  