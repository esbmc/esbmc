typedef struct
{
  char c;
} fam;

typedef struct
{
  short classes[4];
  short e;
  fam f[];
} g;

g h;

int main()
{
  (&h)->classes[h.e];
  return 0;
}
