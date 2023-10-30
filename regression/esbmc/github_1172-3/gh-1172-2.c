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
    do {
        p = rand()%100 + 1;
    } while(!isPrime(p));
    do {
        q = rand()%100 + 1;
    } while(!isPrime(q));
    n = p * q;
    phi = (p-1) * (q-1);

    //Choosing a public key
    do{
        e = rand()%phi;
    }while(gcd(e, phi) != 1);

    //Calculating private key
    int k = 1;
    while(1){
        k = k + phi;
        if(k % e == 0){
            d = k/e;
            break;
        }
    }

    printf("Public Key: (%d, %d)\n", e, n);
    printf("Private Key: (%d, %d)\n", d, n);

    //Encryption
    printf("Enter the plaintext (a single letter): ");
    scanf("%d", &plaintext);
    ciphertext = fmod(pow(plaintext, e), n);
    printf("The ciphertext is: %d\n", ciphertext);

    //Decryption
    plaintext = fmod(pow(ciphertext, d), n);
    printf("The plaintext is: %d\n", plaintext);

    return 0;
}
