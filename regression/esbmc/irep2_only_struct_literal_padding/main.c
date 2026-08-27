/* The same dispatch with the leak freed: the padded literal must not turn a
   correct program into a false alarm either. */
#include <stdlib.h>
#include <string.h>

void c1(void);
void c2(void);

struct command
{
  char *name;
  void (*function)(void);
  char state_needed;
};

int state = 0;
char *g = 0;

const struct command commands[] = {{"c1", c1, 0}, {"c2", c2, 1}};

void parse(char *in)
{
  for (int i = 0; i < 2; i++)
    if (strcmp(commands[i].name, in) == 0 && state >= commands[i].state_needed)
    {
      commands[i].function();
      return;
    }
}

void c1(void)
{
  char *x = malloc(1);
  if (!x)
    return;
  free(g);
  g = x;
  state = 1;
}

void c2(void)
{
  free(g);
  g = 0;
}

int main(void)
{
  parse("c1");
  parse("c1");
  parse("c2");
  return 0;
}
