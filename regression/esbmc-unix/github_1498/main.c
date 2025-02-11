typedef struct {
  int a;
} b;

int main()
{
  ((b *)60000)->a = 0;
}
