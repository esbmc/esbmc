#include <stdio.h>
#include <assert.h>
#include <stddef.h>

// Fib reactor state
typedef struct {
    int N;
    int result;
    int lastResult;
    int secondLastResult;
} Fib;

// Printer reactor state
typedef struct {
    int result;
} Printer;

// Initialize Fib reactor
void fib_init(Fib *self) {
    __ESBMC_requires(self != NULL);
    
    __ESBMC_assigns(self->N, self->result, self->lastResult, self->secondLastResult);
    __ESBMC_ensures(self->N == 0);
    __ESBMC_ensures(self->result == 0);
    __ESBMC_ensures(self->lastResult == 0);
    __ESBMC_ensures(self->secondLastResult == 0);
    
    self->N = 0;
    self->result = 0;
    self->lastResult = 0;
    self->secondLastResult = 0;
}

// Initialize Printer reactor
void printer_init(Printer *self) {
    __ESBMC_requires(self != NULL);
    
    __ESBMC_assigns(self->result);
    __ESBMC_ensures(self->result == 0);
    
    self->result = 0;
}

// Fib timer reaction
void fib_reaction_timer(Fib *self, int *out) {
    __ESBMC_requires(self != NULL);
    __ESBMC_requires(out != NULL);
    
    __ESBMC_assigns(self->result, *out);
    // Case 1: if N < 2, then result == 1
    __ESBMC_ensures(__ESBMC_old(self->N) >= 2 || self->result == 1);
    // Case 2: if N >= 2, then result == lastResult + secondLastResult
    __ESBMC_ensures(__ESBMC_old(self->N) < 2 || self->result == __ESBMC_old(self->lastResult) + __ESBMC_old(self->secondLastResult));
    __ESBMC_ensures(*out == self->result);
    
    if (self->N < 2) {
        self->result = 1;
    } else {
        self->result = self->lastResult + self->secondLastResult;
    }
    *out = self->result;
}

// Fib incrementN reaction
void fib_reaction_incrementN(Fib *self) {
    __ESBMC_requires(self != NULL);
    
    __ESBMC_assigns(self->N);
    __ESBMC_ensures(self->N == __ESBMC_old(self->N) + 1);
    
    self->N += 1;
}

// Fib saveSecondLast reaction
void fib_reaction_saveSecondLast(Fib *self) {
    __ESBMC_requires(self != NULL);
    
    __ESBMC_assigns(self->secondLastResult);
    __ESBMC_ensures(self->secondLastResult == __ESBMC_old(self->lastResult));
    
    self->secondLastResult = self->lastResult;
}

// Fib saveLast reaction
void fib_reaction_saveLast(Fib *self) {
    __ESBMC_requires(self != NULL);
    
    __ESBMC_assigns(self->lastResult);
    __ESBMC_ensures(self->lastResult == __ESBMC_old(self->result));
    
    self->lastResult = self->result;
}

// Printer reaction
void printer_reaction(Printer *self, int in) {
    __ESBMC_requires(self != NULL);
    
    __ESBMC_assigns(self->result);
    __ESBMC_ensures(self->result == in);
    
    self->result = in;
    printf("%d\n", self->result);
}

int main() {
    Fib fib;
    Printer printer;
    int out;
    
    // Initialize reactors
    fib_init(&fib);
    printer_init(&printer);
    
    // Simulate execution for 11 time steps (0 to 10 nsec)
    // Each iteration represents 1 nsec
    for (int time = 0; time <= 10; time++) {
        // Timer reaction fires
        fib_reaction_timer(&fib, &out);
        
        // Printer reaction (corresponds to Fibonacci_printer_reaction_0)
        printer_reaction(&printer, out);
        
        // Logical actions with 0 delay execute in declaration order
        fib_reaction_incrementN(&fib);
        fib_reaction_saveSecondLast(&fib);
        fib_reaction_saveLast(&fib);
        
        // Property verification: G[10 nsec](Fibonacci_printer_reaction_0 ==> Fibonacci_printer_result == 89)
        // At time 10 nsec, when printer reaction executes, result should be 89
        if (time == 10) {
            assert(printer.result == 89);
        }
    }
    
    return 0;
}