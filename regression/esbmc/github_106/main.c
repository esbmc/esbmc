# include <stdio.h>
# include <stdlib.h>
# include <assert.h>

typedef int __nodetype;

typedef struct node {
    __nodetype key;
    struct node *next;
} NODE;

static unsigned int vertices;

int get_graph_size(char *argv[]) {
  unsigned int size=0;
  int a, b;
  FILE *fp;
  fp = fopen(argv[1],"r");
  if (!fp) {
    printf("Failed to open the file %s.\n",argv[1]);
    return -1;
  }
  while (!feof(fp)) {
    fscanf(fp,"%d%d", &a, &b);
    size = a > size ? a : size;
    size = b > size ? b : size;
  }

  ++size;
  printf("graph size: %d\n", size);

  return size;
}

void insert_node(NODE* list[], int a, int b){
    NODE* l = (NODE*)malloc(sizeof(NODE));
    if (list[a] == NULL) {
        l->key = b;
        l->next = NULL;
    } else {
        l->key = b;
        l->next = list[a];
    }
    list[a] = l;
}

void print_adjacent_list(NODE *list[]){
  int i;
  NODE *tmp;
  printf("\nPrinting adjacent list...\n\n");
  for(i=0; i<vertices; i++) {
    if (list[i]!=NULL) {
      printf("(%d) ==> %d ", i, list[i]->key) ;
      tmp = list[i]->next;
      while (tmp != NULL) {
        printf("==> %d  ", tmp->key);
        tmp = tmp->next;
      }
    }
    printf("\n");
  }
}

void print_adjacent_matrix(__nodetype matrix[vertices][vertices]) {
  int i, j;
  printf("\nPrinting adjacent matrix...\n\n");
  for(i=0; i<vertices; i++) {
    for(j=0; j<vertices; j++) {
      printf("%d ", matrix[i][j]);
    }
    printf("\n");
  }
}

int create_adjacent_list(char *argv[], NODE *list[]) {
  int i, a, b, prev_a, prev_b;
  FILE *fp;
  fp = fopen(argv[1],"r");

  for(i=0; i<vertices; i++)
    list[i]=NULL;

  if (!fp) {
    printf("Failed to open the file %s.\n",argv[1]);
    return -1;
  }

  while (!feof(fp)) {
    fscanf(fp,"%d%d", &a, &b);
    if (prev_a != a || prev_b != b) {
          printf("a: %d, b: %d\n", a, b);
      insert_node(list, a, b);
    }
    prev_a = a;
    prev_b = b;
  }
  return 0;
}

int create_adjacent_matrix(char *argv[], __nodetype matrix[vertices][vertices]) {
  int i, j, a, b, prev_a, prev_b;
  FILE *fp;
  fp = fopen(argv[1],"r");

  for(i=0; i<vertices; i++)
    for(j=0; j<vertices; j++)
      matrix[i][j] = 0;

  if (!fp) {
    printf("Failed to open the file %s.\n",argv[1]);
    return -1;
  }

  while (!feof(fp)) {
    fscanf(fp,"%d%d", &a, &b);
    if (prev_a != a || prev_b != b) {
          printf("a: %d, b: %d\n", a, b);
      matrix[a][b]=1;
    }
    prev_a = a;
    prev_b = b;
  }
  return 0;
}

int main(int argc, char *argv[]){

  vertices = get_graph_size(argv);

  NODE *adjacent_list[vertices];
  __nodetype adjacent_matrix[vertices][vertices];

  create_adjacent_list(argv, adjacent_list);
  print_adjacent_list(adjacent_list);

  create_adjacent_matrix(argv, adjacent_matrix);
  print_adjacent_matrix(adjacent_matrix);

  return 0;
}
