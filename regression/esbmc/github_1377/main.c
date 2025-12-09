
//FormAI DATASET v1.0 Category: Pac-Man Game Clone ; Style: Dennis Ritchie
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define BOARD_WIDTH 20
#define BOARD_HEIGHT 15
#define MAX_PELLETS 100

typedef enum
{
  UP,
  DOWN,
  LEFT,
  RIGHT
} Direction;

typedef struct
{
  int x;
  int y;
} Point;

typedef struct
{
  Point position;
  Direction direction;
} Pacman;

typedef struct
{
  Point position;
  int isPowerPellet;
} Pellet;

typedef struct
{
  Pellet pellets[MAX_PELLETS];
  int pelletCount;
} PelletList;

typedef struct
{
  Point position;
  int isOpen;
} Wall;

typedef struct
{
  Wall walls[BOARD_WIDTH][BOARD_HEIGHT];
  PelletList pellets;
  Pacman pacman;
  Direction ghost1Direction;
  Point ghost1Position;
} GameBoard;

/** Initializes a new game board */
void initGameBoard(GameBoard *board)
{
  // Initialize pellets
  board->pellets.pelletCount = 0;
  for (int i = 0; i < BOARD_WIDTH; i++)
  {
    for (int j = 0; j < BOARD_HEIGHT; j++)
    {
      if (
        (i == 0 && j == 0) || (i == BOARD_WIDTH - 1 && j == BOARD_HEIGHT - 1) ||
        (i == 0 && j == BOARD_HEIGHT - 1) || (i == BOARD_WIDTH - 1 && j == 0))
      {
        continue; // Don't put pellets in the corner squares
      }
      Pellet pellet = {{i, j}, rand() % 10 == 0};
      board->pellets.pellets[board->pellets.pelletCount++] = pellet;
    }
  }

  // Initialize walls
  memset(board->walls, 0, sizeof(Wall) * BOARD_WIDTH * BOARD_HEIGHT);
  for (int i = 0; i < BOARD_WIDTH; i++)
  {
    board->walls[i][0].isOpen = 0;
    board->walls[i][BOARD_HEIGHT - 1].isOpen = 0;
  }
  for (int i = 0; i < BOARD_HEIGHT; i++)
  {
    board->walls[0][i].isOpen = 0;
    board->walls[BOARD_WIDTH - 1][i].isOpen = 0;
  }
  for (int i = 4; i < BOARD_WIDTH - 4; i++)
  {
    board->walls[i][3].isOpen = 0;
    board->walls[i][11].isOpen = 0;
  }
  for (int i = 0; i < BOARD_WIDTH; i++)
  {
    board->walls[i][7].isOpen = 0;
  }
  for (int i = 0; i < BOARD_WIDTH; i++)
  {
    board->walls[i][BOARD_HEIGHT - 8].isOpen = 0;
  }
  for (int i = 1; i < BOARD_HEIGHT - 1; i++)
  {
    board->walls[5][i].isOpen = 0;
    board->walls[BOARD_WIDTH - 6][i].isOpen = 0;
  }

  // Initialize pacman
  board->pacman.position.x = 1;
  board->pacman.position.y = 1;
  board->pacman.direction = RIGHT;

  // Initialize ghost1
  board->ghost1Position.x = BOARD_WIDTH - 2;
  board->ghost1Position.y = BOARD_HEIGHT - 2;
  board->ghost1Direction = LEFT;
}

/** Draws the game board to the console */
void drawGameBoard(GameBoard board)
{
  for (int j = 0; j < BOARD_HEIGHT; j++)
  {
    for (int i = 0; i < BOARD_WIDTH; i++)
    {
      if (board.pacman.position.x == i && board.pacman.position.y == j)
      {
        switch (board.pacman.direction)
        {
        case UP:
          printf("^");
          break;
        case DOWN:
          printf("v");
          break;
        case LEFT:
          printf("<");
          break;
        case RIGHT:
          printf(">");
          break;
        }
      }
      else if (board.ghost1Position.x == i && board.ghost1Position.y == j)
      {
        printf("G");
      }
      else if (
        board.walls[i][j].isOpen || i == 0 || i == BOARD_WIDTH - 1 || j == 0 ||
        j == BOARD_HEIGHT - 1)
      {
        printf(".");
      }
      else if (findPellet(board.pellets, i, j) != -1)
      {
        printf("*");
      }
      else
      {
        printf(" ");
      }
    }
    printf("\n");
  }
}

/** Moves pacman in the given direction if possible */
void movePacman(GameBoard *board, Direction direction)
{
  Point newPosition;
  newPosition = board->pacman.position;
  switch (direction)
  {
  case UP:
    newPosition.y--;
    break;
  case DOWN:
    newPosition.y++;
    break;
  case LEFT:
    newPosition.x--;
    break;
  case RIGHT:
    newPosition.x++;
    break;
  }

  if (board->walls[newPosition.x][newPosition.y].isOpen)
  {
    board->pacman.position = newPosition;
    int pelletIndex = findPellet(
      board->pellets, board->pacman.position.x, board->pacman.position.y);
    if (pelletIndex != -1)
    {
      if (board->pellets.pellets[pelletIndex].isPowerPellet)
      {
        // Power pellet
      }
      else
      {
        // Normal pellet
      }
      board->pellets.pellets[pelletIndex].position.x = -1;
      board->pellets.pellets[pelletIndex].position.y = -1;
    }
  }
}

/** Finds the index of the pellet at the given position in the given pellet list */
int findPellet(PelletList pellets, int x, int y)
{
  for (int i = 0; i < pellets.pelletCount; i++)
  {
    if (
      pellets.pellets[i].position.x == x && pellets.pellets[i].position.y == y)
    {
      return i;
    }
  }
  return -1;
}

int main()
{
  srand(time(NULL)); // Seed random number generator

  GameBoard board;
  initGameBoard(&board);

  int score = 0;

  while (1)
  {
    system("cls");
    drawGameBoard(board);
    printf("Score: %d\n", score);

    char input = getchar();

    switch (input)
    {
    case 'w':
      movePacman(&board, UP);
      break;
    case 's':
      movePacman(&board, DOWN);
      break;
    case 'a':
      movePacman(&board, LEFT);
      break;
    case 'd':
      movePacman(&board, RIGHT);
      break;
    case 'q':
      return 0;
    }
  }
}
