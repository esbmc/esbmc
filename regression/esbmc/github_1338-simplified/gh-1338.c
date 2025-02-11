//FormAI DATASET v1.0 Category: Elevator Simulation ; Style: Dennis Ritchie
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define FLOOR_COUNT 2
#define ELEVATOR_CAPACITY 2

// Struct to represent a person
typedef struct {
    int currentFloor;
    int destinationFloor;
} Person;

// Struct to represent elevator status
typedef struct {
    int currentFloor;
    int destinationFloor;
    int totalOccupants;
    Person occupants[ELEVATOR_CAPACITY];
} Elevator;

// Function to simulate a person entering the elevator
void enterElevator(Elevator *elevator, Person *person) {
    elevator->occupants[elevator->totalOccupants++] = *person;
}

// Function to simulate a person exiting the elevator
void exitElevator(Elevator *elevator, int index) {
    for (int i = index; i < elevator->totalOccupants - 1; i++) {
        elevator->occupants[i] = elevator->occupants[i + 1];
    }
    elevator->totalOccupants--;
}

int main() {
    // Seed random number generator
    srand(time(NULL));
    
    // Initialize elevator to be on ground floor with no occupants
    Elevator elevator = {0, 0, 0, {}};
    
    // Initialize array of people waiting on each floor
    Person waiting[FLOOR_COUNT][ELEVATOR_CAPACITY] = {};
    int waitingCounts[FLOOR_COUNT] = {};
    
    // Start elevator simulation loop
    for (int i = 0; i < 1; i++) {
        // Print status of elevator and waiting people
        printf("Time: %d | Elevator floor: %d | Occupants: %d\n", 
               i, elevator.currentFloor, elevator.totalOccupants);
        for (int j = 0; j < FLOOR_COUNT; j++) {
            printf("Floor %d: %d people waiting\n", j + 1, waitingCounts[j]);
        }
        
        // Check if elevator has reached its destination floor
        if (elevator.currentFloor == elevator.destinationFloor) {
            // If elevator is empty, choose a new destination floor at random
            if (elevator.totalOccupants == 0) {
                elevator.destinationFloor = rand() % FLOOR_COUNT;
            }
            // Otherwise, drop off occupants whose destination is the current floor
            else {
                for (int j = 0; j < elevator.totalOccupants; j++) {
                    if (elevator.occupants[j].destinationFloor == elevator.currentFloor) {
                        exitElevator(&elevator, j);
                        j--;
                    }
                }
            }
        }
        // Otherwise, move elevator towards the destination floor
        else {
            elevator.currentFloor += (elevator.destinationFloor > elevator.currentFloor) ? 1 : -1;
        }
        
        // Check each floor to see if people are waiting to get on
        for (int j = 0; j < FLOOR_COUNT; j++) {
            // If elevator is on this floor, try to fill remaining capacity
            if (elevator.currentFloor == j) {
                int spaceLeft = ELEVATOR_CAPACITY - elevator.totalOccupants;
                for (int k = 0; k < spaceLeft && waitingCounts[j] > 0; k++) {
                    enterElevator(&elevator, &waiting[j][0]);
                    for (int l = 0; l < waitingCounts[j] - 1; l++) {
                        waiting[j][l] = waiting[j][l + 1];
                    }
                    waitingCounts[j]--;
                }
            }
            // Otherwise, generate new people at this floor with random destination floors
            else {
                int newPeopleCount = rand() % 4;
                for (int k = 0; k < newPeopleCount && waitingCounts[j] < ELEVATOR_CAPACITY; k++) {
                    Person person = {j, rand() % FLOOR_COUNT};
                    waiting[j][waitingCounts[j]++] = person;
                }
            }
        }
        
        // Wait one second between iterations for readability
        printf("\n");
        sleep(1);
    }
    
    return 0;
}
