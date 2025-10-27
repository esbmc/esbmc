// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2023 Broom team
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdlib.h>
#include <stddef.h> // offsetof

//#define offsetof(type, member) ((size_t)&((type*)0)->member)

struct hlist_node {
        struct hlist_node *next;
};

struct my_item {
    int                 data;
    struct hlist_node   link;
};

int main()
{
    struct my_item *ptr = malloc(sizeof *ptr);

    ptr->data = 42;
    ptr->link.next = NULL;

    struct hlist_node *first = &ptr->link;
    void *tmp = ((void*)first) - offsetof(struct my_item, link);
    struct my_item *now = (struct my_item *) tmp;

    do_data(&(now->data));

    free(ptr);
    return 0;
}

