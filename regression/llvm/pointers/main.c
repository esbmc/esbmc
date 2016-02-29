
int main()
{
  _Bool neh = 2;

  char *e = ("c == a");
  int f = "c == a";
  float a[] = {1,2,3};
  assert(a[0] == 1);
  int b = a;

  int d[5][5];
 (*(&d[0] + 1 + 0))[0]=1;
    *(d+1)[2] = 1;
    *(d+2)[2] = *(d+1)[2];
  int c = *(d+1)[0];

  int g[2][3]={{1,3,0}, {-1,5,9}};
  int century [2][2][2][2][2];

  for (int i = 0; i < 2; i++)
  for (int j = 0; j < 2; j++)
  for (int k = 0; k < 2; k++)
  for (int l = 0; l < 2; l++)
  for (int m = 0; m < 2; m++)
    century[i][j][k][l][m] = 0;

 (*(&century[0][0] + 1 + 1 + 0))[0][0][0]=11;
    *(*century+1)[0][0][0] = 11;
    *(**century+1)[0][0] = 11;


  for (int i = 0; i < 2; i++)
  for (int j = 0; j < 2; j++)
  for (int k = 0; k < 2; k++)
  for (int l = 0; l < 2; l++)
  for (int m = 0; m < 2; m++)
    printf("century[%d][%d][%d][%d][%d] = %d\n", i, j, k, l, m, century[i][j][k][l][m]);

  assert(century[0][0][1][0][0] == 11);
  assert(century[0][1][0][0][0] == 11);
  return 0;
}

