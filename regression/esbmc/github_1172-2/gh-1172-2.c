//FormAI DATASET v1.0 Category: Public-Key Algorithm Implementation ; Style: energetic
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

int isPrime(int n){
    int i;
    for(i = 2; i <= n/2; ++i){
        if(n % i == 0){
            return 0;
        }
    }
    return 1;
}

int gcd(int a, int b){
    int temp;
    while (b != 0){
        temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int main(){
    srand(time(0));
    int p, q, n, phi, e, d, plaintext, ciphertext;
    
    //Generating two prime numbers
    while(!isPrime(p)){
        p = rand()%100 + 1;
    }
    while(!isPrime(q)){
        q = rand()%100 + 1;
    }
    n = p * q;
}
