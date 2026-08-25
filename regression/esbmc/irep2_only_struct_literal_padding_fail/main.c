/* Reduced from github_2335_4. A struct literal reaches this pass with an
   operand per declared member while add_padding has already given the type its
   synthetic ones; left unpadded, the dispatch through commands[i].function
   takes a different flow and the leak on the second c1() goes unreported --
   the flag turned a FAILED into a SUCCESSFUL. The literal's own type is an
   inline pre-padding copy, so the padded layout comes from the tag symbol. */
#include <stdlib.h>
#include <string.h>
void c1(void);
void c2(void);
struct command { char *name; void (*function)(void); char state_needed; };
int state = 0;
char *g = 0;
const struct command commands[] = { {"c1", c1, 0}, {"c2", c2, 1} };
void parse(char *in)
{
  for (int i = 0; i < 2; i++)
    if (strcmp(commands[i].name, in) == 0 && state >= commands[i].state_needed)
    {
      commands[i].function();
      return;
    }
}
void c1(void) { char *x = malloc(1); if (!x) return; g = x; state = 1; }
void c2(void) { free(g); }
int main(void) { parse("c1"); parse("c1"); parse("c2"); return 0; }
