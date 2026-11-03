struct Flags
{
  unsigned int value;
};

volatile struct Flags flag;
int values[2];

int main(void)
{
  values[flag.value] = 1;
}
