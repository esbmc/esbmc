#include <stdlib.h>
#include <string.h>

typedef struct {
    char values[1][1];
} Entry;

typedef struct {
    Entry entries[1];
} Dataset;

void parse_csv_file(Dataset *dataset) {
    int i = 0;
    Entry entry;
    dataset->entries[i] = entry;
}

int main() {
    Dataset dataset;
    parse_csv_file(&dataset);
    return 0;
}
