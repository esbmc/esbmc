// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.6.0;

contract Base {
    int ret;
    int p;
    int q;
    int n;
    int phi;
    int e;
    int d;

    function isPrime(int n) public pure returns (bool) {
        int i;
        for (i = 2; i <= n / 2; ++i) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }

    function gcd(int a, int b) public pure returns (int) {
        int temp;
        while (b != 0) {
            temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    function test(int rand) public {
        //Generating two prime numbers
        p = (rand % 100) + 1;
        while (!isPrime(p)) {
            p = (rand % 100) + 1;
        }
        q = (rand % 100) + 1;
        while (!isPrime(q)) {
            p = (rand % 100) + 1;
        }
        n = p * q;
        phi = (p - 1) * (q - 1);

        //Choosing a public key
        e = rand % phi;
        while (gcd(e, phi) != 1) {
            p = (rand % 100) + 1;
        }

        //Calculating private key
        int k = 1;
        while (true) {
            k = k + phi;
            if (k % e == 0) {
                d = k / e;
            }
        }
    }
}
