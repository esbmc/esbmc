/*
 * Concurrency Verification Example
 *
 * Demonstrates ESBMC's concurrency verification capabilities:
 * - Deadlock detection
 * - Data race detection
 * - Lock ordering violations
 * - Atomicity checks
 *
 * Run with: esbmc concurrent.c --deadlock-check --data-races-check --context-bound 2 --unwind 5
 */

#include <pthread.h>
#include <assert.h>

// Non-deterministic input
int __ESBMC_nondet_int(void);
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);

// Atomic operations
void __ESBMC_atomic_begin(void);
void __ESBMC_atomic_end(void);

/* ============================================
 * Example 1: Simple Data Race
 * ============================================ */

int shared_counter = 0;

void *increment_thread_unsafe(void *arg) {
    // Data race: multiple threads access shared_counter without synchronization
    shared_counter++;
    return NULL;
}

void data_race_example(void) {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, increment_thread_unsafe, NULL);
    pthread_create(&t2, NULL, increment_thread_unsafe, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // Result is undefined due to data race
    // ESBMC with --data-races-check will detect this
}

/* ============================================
 * Example 2: Safe Counter with Mutex
 * ============================================ */

int safe_counter = 0;
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void *increment_thread_safe(void *arg) {
    pthread_mutex_lock(&counter_mutex);
    safe_counter++;
    pthread_mutex_unlock(&counter_mutex);
    return NULL;
}

void safe_counter_example(void) {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, increment_thread_safe, NULL);
    pthread_create(&t2, NULL, increment_thread_safe, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // With mutex protection, counter will be exactly 2
    __ESBMC_assert(safe_counter == 2, "Counter incremented correctly");
}

/* ============================================
 * Example 3: Deadlock - Circular Wait
 * ============================================ */

pthread_mutex_t mutex_a = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_b = PTHREAD_MUTEX_INITIALIZER;
int resource_a = 0;
int resource_b = 0;

void *thread_ab(void *arg) {
    pthread_mutex_lock(&mutex_a);
    // Thread 1 holds A, wants B
    pthread_mutex_lock(&mutex_b);

    resource_a = 1;
    resource_b = 1;

    pthread_mutex_unlock(&mutex_b);
    pthread_mutex_unlock(&mutex_a);
    return NULL;
}

void *thread_ba(void *arg) {
    pthread_mutex_lock(&mutex_b);
    // Thread 2 holds B, wants A - DEADLOCK!
    pthread_mutex_lock(&mutex_a);

    resource_a = 2;
    resource_b = 2;

    pthread_mutex_unlock(&mutex_a);
    pthread_mutex_unlock(&mutex_b);
    return NULL;
}

void deadlock_example(void) {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, thread_ab, NULL);
    pthread_create(&t2, NULL, thread_ba, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // ESBMC with --deadlock-check will detect the potential deadlock
}

/* ============================================
 * Example 4: Safe Lock Ordering
 * ============================================ */

void *thread_ordered_1(void *arg) {
    // Always acquire locks in same order: A then B
    pthread_mutex_lock(&mutex_a);
    pthread_mutex_lock(&mutex_b);

    resource_a = 1;
    resource_b = 1;

    pthread_mutex_unlock(&mutex_b);
    pthread_mutex_unlock(&mutex_a);
    return NULL;
}

void *thread_ordered_2(void *arg) {
    // Same order: A then B - no deadlock possible
    pthread_mutex_lock(&mutex_a);
    pthread_mutex_lock(&mutex_b);

    resource_a = 2;
    resource_b = 2;

    pthread_mutex_unlock(&mutex_b);
    pthread_mutex_unlock(&mutex_a);
    return NULL;
}

void safe_lock_ordering_example(void) {
    pthread_t t1, t2;

    resource_a = 0;
    resource_b = 0;

    pthread_create(&t1, NULL, thread_ordered_1, NULL);
    pthread_create(&t2, NULL, thread_ordered_2, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // No deadlock with consistent lock ordering
}

/* ============================================
 * Example 5: Atomicity Violation
 * ============================================ */

int balance = 100;
pthread_mutex_t balance_mutex = PTHREAD_MUTEX_INITIALIZER;

void *withdraw_nonatomic(void *arg) {
    int amount = 50;

    pthread_mutex_lock(&balance_mutex);
    int current = balance;
    pthread_mutex_unlock(&balance_mutex);

    // Check-then-act race: balance can change between check and update
    if (current >= amount) {
        pthread_mutex_lock(&balance_mutex);
        balance = balance - amount;  // May go negative!
        pthread_mutex_unlock(&balance_mutex);
    }
    return NULL;
}

void atomicity_violation_example(void) {
    pthread_t t1, t2;

    balance = 100;

    pthread_create(&t1, NULL, withdraw_nonatomic, NULL);
    pthread_create(&t2, NULL, withdraw_nonatomic, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // Balance could be negative due to atomicity violation
    // ESBMC can detect this
}

/* ============================================
 * Example 6: Atomic Withdraw (Safe)
 * ============================================ */

int safe_balance = 100;

void *withdraw_atomic(void *arg) {
    int amount = 50;

    pthread_mutex_lock(&balance_mutex);
    // Atomic check-and-update
    if (safe_balance >= amount) {
        safe_balance = safe_balance - amount;
    }
    pthread_mutex_unlock(&balance_mutex);

    return NULL;
}

void safe_withdraw_example(void) {
    pthread_t t1, t2;

    safe_balance = 100;

    pthread_create(&t1, NULL, withdraw_atomic, NULL);
    pthread_create(&t2, NULL, withdraw_atomic, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // Balance is always non-negative
    __ESBMC_assert(safe_balance >= 0, "Balance non-negative");
    __ESBMC_assert(safe_balance == 0 || safe_balance == 50, "Valid final balance");
}

/* ============================================
 * Example 7: Producer-Consumer
 * ============================================ */

#define BUFFER_SIZE 5

int buffer[BUFFER_SIZE];
int count = 0;
int in_idx = 0;
int out_idx = 0;
pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;

void *producer(void *arg) {
    int item = __ESBMC_nondet_int();
    __ESBMC_assume(item > 0);

    pthread_mutex_lock(&buffer_mutex);
    if (count < BUFFER_SIZE) {
        buffer[in_idx] = item;
        in_idx = (in_idx + 1) % BUFFER_SIZE;
        count++;
    }
    pthread_mutex_unlock(&buffer_mutex);

    return NULL;
}

void *consumer(void *arg) {
    int item = 0;

    pthread_mutex_lock(&buffer_mutex);
    if (count > 0) {
        item = buffer[out_idx];
        out_idx = (out_idx + 1) % BUFFER_SIZE;
        count--;
        __ESBMC_assert(item > 0, "Consumed valid item");
    }
    pthread_mutex_unlock(&buffer_mutex);

    return NULL;
}

void producer_consumer_example(void) {
    pthread_t p1, p2, c1, c2;

    count = 0;
    in_idx = 0;
    out_idx = 0;

    pthread_create(&p1, NULL, producer, NULL);
    pthread_create(&p2, NULL, producer, NULL);
    pthread_create(&c1, NULL, consumer, NULL);
    pthread_create(&c2, NULL, consumer, NULL);

    pthread_join(p1, NULL);
    pthread_join(p2, NULL);
    pthread_join(c1, NULL);
    pthread_join(c2, NULL);

    // Count is always in valid range
    __ESBMC_assert(count >= 0 && count <= BUFFER_SIZE, "Count in valid range");
}

/* ============================================
 * Example 8: Using ESBMC Atomic Intrinsics
 * ============================================ */

int atomic_counter = 0;

void *atomic_increment(void *arg) {
    __ESBMC_atomic_begin();
    // This block executes atomically
    int temp = atomic_counter;
    temp++;
    atomic_counter = temp;
    __ESBMC_atomic_end();

    return NULL;
}

void esbmc_atomic_example(void) {
    pthread_t t1, t2, t3;

    atomic_counter = 0;

    pthread_create(&t1, NULL, atomic_increment, NULL);
    pthread_create(&t2, NULL, atomic_increment, NULL);
    pthread_create(&t3, NULL, atomic_increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);

    // With atomic blocks, counter is exactly 3
    __ESBMC_assert(atomic_counter == 3, "Atomic counter correct");
}

/* ============================================
 * Example 9: Reader-Writer Pattern
 * ============================================ */

int shared_data = 0;
int readers = 0;
pthread_mutex_t rw_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t readers_mutex = PTHREAD_MUTEX_INITIALIZER;

void *reader(void *arg) {
    pthread_mutex_lock(&readers_mutex);
    readers++;
    if (readers == 1) {
        pthread_mutex_lock(&rw_mutex);  // First reader locks
    }
    pthread_mutex_unlock(&readers_mutex);

    // Read shared data
    int value = shared_data;
    __ESBMC_assert(value >= 0, "Read non-negative value");

    pthread_mutex_lock(&readers_mutex);
    readers--;
    if (readers == 0) {
        pthread_mutex_unlock(&rw_mutex);  // Last reader unlocks
    }
    pthread_mutex_unlock(&readers_mutex);

    return NULL;
}

void *writer(void *arg) {
    pthread_mutex_lock(&rw_mutex);
    shared_data = __ESBMC_nondet_int();
    __ESBMC_assume(shared_data >= 0);
    pthread_mutex_unlock(&rw_mutex);

    return NULL;
}

void reader_writer_example(void) {
    pthread_t r1, r2, w1;

    shared_data = 0;
    readers = 0;

    pthread_create(&r1, NULL, reader, NULL);
    pthread_create(&w1, NULL, writer, NULL);
    pthread_create(&r2, NULL, reader, NULL);

    pthread_join(r1, NULL);
    pthread_join(w1, NULL);
    pthread_join(r2, NULL);
}

int main(void) {
    // Safe examples (pass verification):
    safe_counter_example();
    safe_lock_ordering_example();
    safe_withdraw_example();
    producer_consumer_example();
    esbmc_atomic_example();
    reader_writer_example();

    // Unsafe examples (fail verification - uncomment to test):
    // data_race_example();
    // deadlock_example();
    // atomicity_violation_example();

    return 0;
}
