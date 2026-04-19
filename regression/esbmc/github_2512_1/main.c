// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2023 Broom team
//
// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Double linked lists with a single pointer list head.
 * Functions which create, traverse, and destroy list forwards
 * based on file from Linux Kernel (include/linux/list.h)
 */

#include <stdlib.h>
#include <stddef.h> // offsetof
#define typeof __typeof__

extern int __VERIFIER_nondet_int(void);
// void __VERIFIER_plot(const char *name, ...);
#define random() __VERIFIER_nondet_int()

struct hlist_head {
        struct hlist_node *first;
};

struct hlist_node {
        struct hlist_node *next, **pprev;
};

/*
 * Double linked lists with a single pointer list head.
 * Mostly useful for hash tables where the two pointer list head is
 * too wasteful.
 * You lose the ability to access the tail in O(1).
 */

#define HLIST_HEAD_INIT { .first = NULL }
#define HLIST_HEAD(name) struct hlist_head name = {  .first = NULL }
#define INIT_HLIST_HEAD(ptr) ((ptr)->first = NULL)
void INIT_HLIST_NODE(struct hlist_node *h)
{
        h->next = NULL;
        h->pprev = NULL;
}

/**
 * hlist_add_head - add a new entry at the beginning of the hlist
 * @n: new entry to be added
 * @h: hlist head to add it after
 *
 * Insert a new entry after the specified head.
 * This is good for implementing stacks.
 */
void hlist_add_head(struct hlist_node *n, struct hlist_head *h)
{
        struct hlist_node *first = h->first;
        n->next = first;
        if (first)
                first->pprev = &n->next;
        h->first = n;
        n->pprev = &h->first;
}

void __hlist_del(struct hlist_node *n)
{
        struct hlist_node *next = n->next;
        struct hlist_node **pprev = n->pprev;

        *pprev = next;
        if (next)
                next->pprev = pprev;
}

/**
 * hlist_del - Delete the specified hlist_node from its list
 * @n: Node to delete.
 *
 * Note that this function leaves the node in hashed state.  Use
 * hlist_del_init() or similar instead to unhash @n.
 */
void hlist_del(struct hlist_node *n)
{
        __hlist_del(n);
        n->next = (void *) 0;
        n->pprev = (void *) 0;
}

/**
 * container_of - cast a member of a structure out to the containing structure
 * @ptr:        the pointer to the member.
 * @type:       the type of the container struct this is embedded in.
 * @member:     the name of the member within the struct.
 *
 * WARNING: any const qualifier of @ptr is lost.
 */
#define container_of(ptr, type, member) ({                              \
        void *__mptr = (void *)(ptr);                                           \
        ((type *)(__mptr - offsetof(type, member))); })

#define hlist_entry(ptr, type, member) container_of(ptr,type,member)

#define hlist_entry_safe(ptr, type, member) \
        ({ typeof(ptr) ____ptr = (ptr); \
           ____ptr ? hlist_entry(____ptr, type, member) : NULL; \
        })

/**
 * hlist_for_each_entry - iterate over list of given type
 * @pos:        the type * to use as a loop cursor.
 * @head:       the head for your list.
 * @member:     the name of the hlist_node within the struct.
 */
#define hlist_for_each_entry(pos, head, member)                         \
        for (pos = hlist_entry_safe((head)->first, typeof(*(pos)), member);\
             pos;                                                       \
             pos = hlist_entry_safe((pos)->member.next, typeof(*(pos)), member))

/**
 * hlist_for_each_entry_safe - iterate over list of given type safe against removal of list entry
 * @pos:        the type * to use as a loop cursor.
 * @n:          a &struct hlist_node to use as temporary storage
 * @head:       the head for your list.
 * @member:     the name of the hlist_node within the struct.
 */
#define hlist_for_each_entry_safe(pos, n, head, member)                 \
        for (pos = hlist_entry_safe((head)->first, typeof(*pos), member);\
             pos && ({ n = pos->member.next; 1; });                     \
             pos = hlist_entry_safe(n, typeof(*pos), member))

/************************************************************
   Test functions
************************************************************/

struct my_item {
    int                 data;
    struct hlist_node   link;
};

void do_data(int *data)
{
    *data;
}

struct hlist_head *create()
{
    struct hlist_head *head=malloc(sizeof(struct hlist_head));
    INIT_HLIST_HEAD(head);
    while(random()) {
        struct my_item *ptr = malloc(sizeof *ptr);
        INIT_HLIST_NODE(&ptr->link);
        ptr->data = __VERIFIER_nondet_int();
        hlist_add_head(&ptr->link, head);
    }

    return head;
}

void loop(struct hlist_head *head)
{
    struct my_item *now;
    hlist_for_each_entry(now, head, link) {
        do_data(&(now->data));
    }
}

void destroy(struct hlist_head *head)
{
    struct my_item *now;
    struct hlist_node *tmp;
    hlist_for_each_entry_safe(now, tmp, head, link) {
        hlist_del(&now->link);
        free(now);
    }
    free(head);
}

int main()
{
    struct hlist_head *l = create();
    // __VERIFIER_plot("create");
    loop(l);
    destroy(l);
    return 0;
}
