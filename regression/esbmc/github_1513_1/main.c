//FormAI DATASET v1.0 Category: Hotel Management System ; Style: relaxed
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Structure for each room's details.
struct Room {
    char name[20];
    int roomNumber;
    int occupancy;
};

// Function to print available rooms.
void printRooms(struct Room rooms[], int numRooms) {
    printf("List of available rooms: \n");
    for (int i = 0; i < numRooms; i++) {
        if (rooms[i].occupancy == 0) {
            printf("Room %d: %s \n", rooms[i].roomNumber, rooms[i].name);
        }
    }
}

int main() {
    int numRooms;
    printf("Enter the number of rooms in the hotel: ");
    scanf("%d", &numRooms);
    struct Room rooms[numRooms];

    // Get the details of each room from the user.
    for (int i = 0; i < numRooms; i++) {
        printf("Enter the name and room number for room %d: ", i+1);
        scanf("%s %d", &rooms[i].name, &rooms[i].roomNumber);
        rooms[i].occupancy = 0; // Set to 0 initially (i.e., room is not occupied).
    }

    // Main loop of the program.
    int choice = 0;
    while (choice != 4) {
        printf("\nMENU: \n");
        printf("1. Check room availability \n");
        printf("2. Book a room \n");
        printf("3. Check out of a room \n");
        printf("4. Exit \n");
        printf("Enter your choice (1-4): ");
        scanf("%d", &choice);

        switch(choice) {
            case 1:
                printRooms(rooms, numRooms);
                break;

            case 2:
                printf("Enter the room number you wish to book: ");
                int roomToBook;
                scanf("%d", &roomToBook);

                // Find the room with the given room number.
                int indexToBook;
                for (int i = 0; i < numRooms; i++) {
                    if (rooms[i].roomNumber == roomToBook) {
                        indexToBook = i;
                        break;
                    }
                }

                // If the room is available, book it.
                if (rooms[indexToBook].occupancy == 0) {
                    rooms[indexToBook].occupancy = 1;
                    printf("Room %d has been booked for you. Enjoy your stay!\n", rooms[indexToBook].roomNumber);
                } else {
                    printf("Sorry, the room is already occupied. Please choose a different room.\n");
                }
                break;

            case 3:
                printf("Enter the room number you wish to check out of: ");
                int roomToCheckOut;
                scanf("%d", &roomToCheckOut);

                // Find the room with the given room number.
                int indexToCheckOut;
                for (int i = 0; i < numRooms; i++) {
                    if (rooms[i].roomNumber == roomToCheckOut) {
                        indexToCheckOut = i;
                        break;
                    }
                }

                // If the room is occupied, check out and print the bill.
                if (rooms[indexToCheckOut].occupancy == 1) {
                    rooms[indexToCheckOut].occupancy = 0;
                    int bill = rand() % 500 + 500; // Generate random bill amount between $500-$1000.
                    printf("Thank you for your stay. Your bill is $%d.\n", bill);
                } else {
                    printf("Sorry, the room is not occupied. Please choose a different room.\n");
                }
                break;

            case 4:
                printf("Thank you for using the hotel management system!\n");
                break;

            default:
                printf("Invalid choice. Please try again.\n");
        }
    }

    return 0;
}

