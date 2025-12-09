/*
 * This is a miniaturized example of a complex memory allocation mechanism
 * found in bftpd v4.6
 * https://sourceforge.net/projects/bftpd/
 *
 * The memsafety verification complexity comes from obscure flows in the code
 * due to the use of function pointers, global variables,
 * and a global state machine.
 */

#include <stdlib.h>
#include <string.h>

void c1();
void c2();

enum {
        STATE_1,STATE_2
};

struct command {
        char *name;
        void (*function)();
        char state_needed;
};

int state = STATE_1;
char *global = 0;

const struct command commands[] = {
                {"c1",c1,STATE_1},
                {"c2",c2,STATE_2}
};

void parse_input(char *input) {
        for(int i = 0; i < 2; i++) {
                if(strcmp(commands[i].name,input) == 0) {
                        if(state >= commands[i].state_needed) {
                                commands[i].function();
                                return;
                        }
                }
        }
}

void c1() {
        char *x = (char *)malloc(sizeof(char));
        if(!x) {
                // out of memory
                return;
        }

        if(global) {
                // free(global); // memory-leak in next line if c1 is executed twice
        }
        global = x;
        state = STATE_2;
}

void c2() {
        char *y = 0;
        if(!y || !global) {
                free(global);
                return;
        }
        free(y);
        free(global);
        state = STATE_1;
}

int main(void) {
        parse_input("c1");
        parse_input("c1");
        parse_input("c2");
        return 0;
}

