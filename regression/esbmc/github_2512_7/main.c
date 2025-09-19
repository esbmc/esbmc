#include <stdlib.h>
#include <stddef.h>

struct list_node {
    struct list_node *next;
    struct list_node *prev;
};

struct list_item {
    int id;
    struct list_node node;
    char name[32];
};

int main() {
    struct list_item *item1 = malloc(sizeof(struct list_item));
    struct list_item *item2 = malloc(sizeof(struct list_item));
    
    item1->id = 1;
    item2->id = 2;
    
    item1->node.next = &item2->node;
    item2->node.prev = &item1->node;
    item1->node.prev = NULL;
    item2->node.next = NULL;
    
    struct list_node *node_ptr = &item1->node;
    struct list_node *next_node = node_ptr->next;
    void *tmp = ((void*)next_node) - offsetof(struct list_item, node);
    struct list_item *next_item = (struct list_item*)tmp;
    
    int id = next_item->id;
    
    free(item1);
    free(item2);
    return 0;
}
