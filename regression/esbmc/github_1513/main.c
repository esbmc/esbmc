//FormAI DATASET v1.0 Category: Automated Fortune Teller ; Style: multiplayer
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX_NAME_LENGTH 30
#define MAX_MESSAGE_LENGTH 100

struct Player {
    char name[MAX_NAME_LENGTH];
    char fortune[MAX_MESSAGE_LENGTH];
};

void generate_fortune(struct Player *player);

int main() {
    srand(time(NULL)); //set the random seed based on the current time
    int num_players;
    printf("Welcome to the Automated Fortune Teller!\nHow many players will be playing today? ");
    fflush(stdout); //force printing to the console before scanf
    scanf("%d", &num_players);

    struct Player players[num_players];

    for (int i = 0; i < num_players; i++) {
        printf("Player %d, what is your name? ", i + 1);
        fflush(stdout);
        scanf("%s", players[i].name);
        generate_fortune(&players[i]);
        printf("Hello %s, your fortune is: %s\n", players[i].name, players[i].fortune);
    }

    return 0; //exit program
}

void generate_fortune(struct Player *player) {
    char *fortunes[7] = {"Good things come to those who wait.", 
                         "Your future is looking bright!",
                         "The path to success is never easy, but it will be worth it.",
                         "You will soon receive a pleasant surprise.",
                         "Take a moment to reflect, it will benefit you in the long run.",
                         "Be patient with yourself and those around you.",
                         "Your hard work will pay off in due time."};
    //generate a random index to select a fortune
    int index = rand() % 7;
    strcpy(player->fortune, fortunes[index]); //copy the fortune to the player's struct
}

