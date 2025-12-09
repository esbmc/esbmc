//FormAI DATASET v1.0 Category: Data mining ; Style: Dennis Ritchie
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_LINE_LENGTH 256
#define MAX_ENTRIES 1000
#define MAX_CHARS_PER_ENTRY 50
#define MAX_VALUES_PER_ENTRY 5

typedef struct {
    char values[MAX_VALUES_PER_ENTRY][MAX_CHARS_PER_ENTRY];
} Entry;

typedef struct {
    char labels[MAX_VALUES_PER_ENTRY][MAX_CHARS_PER_ENTRY];
    int num_entries;
    Entry entries[MAX_ENTRIES];
} Dataset;

// function to parse a CSV file and populate a Dataset struct
void parse_csv_file(char *filename, Dataset *dataset) {
    FILE *fp;
    char line[MAX_LINE_LENGTH];
    char *token;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Could not open file %s.", filename);
        exit(EXIT_FAILURE);
    }

    // read the first row as labels
    fgets(line, MAX_LINE_LENGTH, fp);
    token = strtok(line, ",");
    int i = 0;
    while (token != NULL) {
        strcpy(dataset->labels[i], token);
        token = strtok(NULL, ",");
        i++;
    }

    // read the rest of the file as entries
    i = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {
        Entry entry;
        token = strtok(line, ",");
        int j = 0;
        while (token != NULL) {
            strcpy(entry.values[j], token);
            token = strtok(NULL, ",");
            j++;
        }
        dataset->entries[i] = entry;
        i++;
    }
    dataset->num_entries = i;

    fclose(fp);
}

// function to print the dataset
void print_dataset(Dataset dataset) {
    printf("Num Entries: %d\n", dataset.num_entries);
    printf("Labels: ");
    for (int i = 0; i < MAX_VALUES_PER_ENTRY; i++) {
        printf("%s, ", dataset.labels[i]);
    }
    printf("\n");
    for (int i = 0; i < dataset.num_entries; i++) {
        for (int j = 0; j < MAX_VALUES_PER_ENTRY; j++) {
            printf("%s, ", dataset.entries[i].values[j]);
        }
        printf("\n");
    }
}

int main() {
    Dataset dataset;
    parse_csv_file("data.csv", &dataset);
    print_dataset(dataset);
    return 0;
}
