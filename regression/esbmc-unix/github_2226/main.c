//Falcon-180B DATASET v1.0 Category: Greedy Algorithms ; Style: distributed
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CITIES 100
#define MAX_NAME_LENGTH 20
#define MAX_ROUTES 1000
#define INF 999999

typedef struct
{
  int id;
  char name[MAX_NAME_LENGTH];
  int x, y;
} City;

typedef struct
{
  int start, end;
  int weight;
} Route;

int n, m, start, end, ans = 0;
City cities[MAX_CITIES];
Route routes[MAX_ROUTES];

void init()
{
  scanf("%d%d", &n, &m);
  for (int i = 0; i < n; i++)
  {
    scanf("%s%d%d", cities[i].name, &cities[i].x, &cities[i].y);
  }
  for (int i = 0; i < m; i++)
  {
    scanf("%d%d%d", &routes[i].start, &routes[i].end, &routes[i].weight);
  }
}

void solve()
{
  int dist[n][n];
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      dist[i][j] = INF;
    }
  }
  for (int i = 0; i < n; i++)
  {
    dist[i][i] = 0;
  }
  for (int i = 0; i < m; i++)
  {
    int u = routes[i].start, v = routes[i].end;
    dist[u][v] = routes[i].weight;
    dist[v][u] = routes[i].weight;
  }
  for (int k = 0; k < n; k++)
  {
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        if (
          dist[i][k] != INF && dist[k][j] != INF &&
          dist[i][j] > dist[i][k] + dist[k][j])
        {
          dist[i][j] = dist[i][k] + dist[k][j];
        }
      }
    }
  }
  for (int i = 0; i < n; i++)
  {
    ans += dist[i][start];
  }
}

void output()
{
  printf("%d\n", ans);
}

int main()
{
  init();
  solve();
  output();
  return 0;
}
