//FormAI DATASET v0.1 Category: Building a XML Parser ; Style: cheerful
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* tag;
    char* value;
} XMLNode;

void parseXML(char* xml) {
    char* ptr = xml;

    while (*ptr != '\0') {
        if (*ptr == '<') {
            XMLNode node;
            char* start_tag = ptr;

            // get the node tag
            while (*ptr != '>' && *ptr != ' ') {
                ptr++;
            }
            node.tag = (char*) malloc((ptr - start_tag) + 1);
            strncpy(node.tag, start_tag+1, (ptr - start_tag) - 1);
            node.tag[ptr - start_tag - 1] = '\0';

            //get the node value
            start_tag = ptr + 1;
            while (*start_tag == ' ') {
                start_tag++;
            }
            if (*start_tag != '<') {
                ptr = start_tag;
                while (*ptr != '<') {
                    ptr++;
                }
                node.value = (char*) malloc((ptr - start_tag) + 1);
                strncpy(node.value, start_tag, ptr - start_tag);
                node.value[ptr - start_tag] = '\0';

                printf("Tag: %s -> Value: %s\n", node.tag, node.value);
            } else {
                printf("Tag: %s -> No Value\n", node.tag);
            }

            free(node.tag);
            free(node.value);
        }
        ptr++;
    }
}

int main(void) {
    char* xml = "<root><name>XML Parser</name><author>John Doe</author><version>1.0.0</version></root>";
    parseXML(xml);
    return EXIT_SUCCESS;
}
