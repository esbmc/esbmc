//FormAI DATASET v0.1 Category: Resume Parsing System ; Style: multiplayer
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Player struct to hold player information
typedef struct Player {
    char name[20];
    int score;
} Player;

// Function to display player information
void displayPlayer(Player p) {
    printf("Player: %s\n", p.name);
    printf("Score: %d\n", p.score);
}

// Function to compare player scores for sorting
int compare(Player a, Player b) {
    return b.score - a.score;
}

int main() {
    int num_players;
    printf("Enter number of players:");
    scanf("%d", &num_players);
    Player* players = (Player*)malloc(num_players * sizeof(Player));

    // Input player information
    for (int i = 0; i < num_players; i++) {
        printf("\nEnter details of player %d:\n", i+1);
        printf("Name: ");
        scanf("%s", players[i].name);
        printf("Score: ");
        scanf("%d", &players[i].score);
    }

    // Sort players based on score
    qsort(players, num_players, sizeof(Player), (void*)compare);

    // Display sorted player information
    printf("\nSorted player information:\n");
    for (int i = 0; i < num_players; i++) {
        displayPlayer(players[i]);
    }

    // Free memory allocated for players
    free(players);

    return 0;
}
