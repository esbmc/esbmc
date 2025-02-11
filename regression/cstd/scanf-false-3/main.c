//FormAI DATASET v1.0 Category: Ebook reader ; Style: futuristic
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_BOOKS 1000
#define MAX_TITLE_LEN 100
#define MAX_AUTHOR_LEN 50

const char* LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi vitae sodales lorem. Curabitur suscipit nec magna vel sagittis.";

typedef struct book_t {
    char title[MAX_TITLE_LEN];
    char author[MAX_AUTHOR_LEN];
    char* content;
    int num_pages;
} Book;

typedef struct ebook_reader_t {
    Book* shelf[MAX_BOOKS];
    int num_books;
    Book* current_book;
} EbookReader;

void print_menu() {
    printf("\n********** Ebook Reader **********\n");
    printf("[1] Add book to shelf\n");
    printf("[2] Open book\n");
    printf("[3] Close book\n");
    printf("[4] Show book shelf\n");
    printf("[5] Exit\n");
}

void add_book_to_shelf(EbookReader* reader) {
    if (reader->num_books >= MAX_BOOKS) {
        printf("Error: maximum number of books in shelf reached.\n");
        return;
    }

    Book* book_ptr = malloc(sizeof(Book));
    printf("Enter book title: ");
    fgets(book_ptr->title, MAX_TITLE_LEN, stdin);
    book_ptr->title[strcspn(book_ptr->title, "\n")] = '\0';
    printf("Enter book author: ");
    fgets(book_ptr->author, MAX_AUTHOR_LEN, stdin);
    book_ptr->author[strcspn(book_ptr->author, "\n")] = '\0';
    printf("Enter number of pages: ");
    scanf("%d", &book_ptr->num_pages);
    getchar(); // consume newline character
    book_ptr->content = malloc(book_ptr->num_pages * strlen(LOREM_IPSUM) + 1);
    strcpy(book_ptr->content, "");
    for (int i = 0; i < book_ptr->num_pages; i++) {
        strcat(book_ptr->content, LOREM_IPSUM);
    }
    reader->shelf[reader->num_books] = book_ptr;
    reader->num_books++;
}

void open_book(EbookReader* reader, int book_index) {
    if (book_index < 0 || book_index >= reader->num_books) {
        printf("Error: invalid book index.\n");
        return;
    }

    reader->current_book = reader->shelf[book_index];
    printf("Opened book: %s by %s\n", reader->current_book->title, reader->current_book->author);
}

void close_book(EbookReader* reader) {
    reader->current_book = NULL;
    printf("Closed book.\n");
}

void show_book_shelf(EbookReader* reader) {
    printf("\nBook shelf:\n");
    for (int i = 0; i < reader->num_books; i++) {
        printf("[%d] %s by %s (%d pages)\n", i, reader->shelf[i]->title, reader->shelf[i]->author, reader->shelf[i]->num_pages);
    }
}

int main() {
    EbookReader reader;
    reader.num_books = 0;
    reader.current_book = NULL;

    while (1) {
        print_menu();
        int choice;
        printf("Enter choice: ");
        scanf("%d", &choice);
        getchar(); // consume newline character

        switch (choice) {
            case 1:
                add_book_to_shelf(&reader);
                break;
            case 2:
                if (!reader.current_book) {
                    printf("Error: no book opened.\n");
                    break;
                }
                int book_index;
                printf("Enter book index: ");
                scanf("%d", &book_index);
                getchar(); // consume newline character
                open_book(&reader, book_index);
                break;
            case 3:
                close_book(&reader);
                break;
            case 4:
                show_book_shelf(&reader);
                break;
            case 5:
                printf("Exiting ebook reader...\n");
                for (int i = 0; i < reader.num_books; i++) {
                    free(reader.shelf[i]->content);
                    free(reader.shelf[i]);
                }
                exit(0);
            default:
                printf("Error: invalid choice.\n");
                break;
        }
    }

    return 0;
}