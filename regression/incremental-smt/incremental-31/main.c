#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/*
Initial balance: $1000
Thread 140736: Checking balance... $1000 available
Thread 140737: Checking balance... $1000 available
Thread 140736: Withdrew $800, new balance: $200
Thread 140737: Withdrew $600, new balance: -$400
Final balance: -$400
*/

pthread_mutex_t account_mutex = PTHREAD_MUTEX_INITIALIZER;

int balance = 1000;  // Shared bank account balance

void* withdraw(void* arg) {
    int amount = *(int*)arg;

    pthread_mutex_lock(&account_mutex);
    // Check if we have enough money
    if (balance >= amount) {
        printf("Thread %ld: Checking balance... $%d available\n",
               pthread_self(), balance);

        // Simulate some processing time (ATM communication, etc.)
        usleep(100000);  // 100ms delay

        // Withdraw the money
        balance -= amount;
        printf("Thread %ld: Withdrew $%d, new balance: $%d\n",
               pthread_self(), amount, balance);
    } else {
        printf("Thread %ld: Insufficient funds for $%d withdrawal\n",
               pthread_self(), amount);
    }
    pthread_mutex_unlock(&account_mutex);
    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    int amount1 = 800;
    int amount2 = 600;

    printf("Initial balance: $%d\n", balance);

    // Two people trying to withdraw money simultaneously
    pthread_create(&thread1, NULL, withdraw, &amount1);
    pthread_create(&thread2, NULL, withdraw, &amount2);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    printf("Final balance: $%d\n", balance);

    return 0;
}