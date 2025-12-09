//FormAI DATASET v1.0 Category: Music Library Management System ; Style: Ken Thompson
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ALBUMS 100
#define MAX_SONGS 1000

struct Song {
    char name[50];
    char artist[50];
    int duration;
};

struct Album {
    char name[50];
    char artist[50];
    int year;
    struct Song songs[MAX_SONGS];
    int num_songs;
};

struct Library {
    struct Album albums[MAX_ALBUMS];
    int num_albums;
};

void add_album(struct Library *library, char *name, char *artist, int year) {
    int num_albums = library->num_albums;
    if (num_albums < MAX_ALBUMS) {
        struct Album *album = &library->albums[num_albums];
        strcpy(album->name, name);
        strcpy(album->artist, artist);
        album->year = year;
        album->num_songs = 0;
        library->num_albums++;
    }
}

void add_song(struct Library *library, char *album_name, char *song_name, char *artist_name, int duration) {
    struct Album *album = NULL;
    for (int i = 0; i < library->num_albums; i++) {
        if (strcmp(library->albums[i].name, album_name) == 0) {
            album = &library->albums[i];
            break;
        }
    }
    if (album != NULL) {
        struct Song *song = &album->songs[album->num_songs];
        strcpy(song->name, song_name);
        strcpy(song->artist, artist_name);
        song->duration = duration;
        album->num_songs++;
    }
}

void print_library(struct Library *library) {
    for (int i = 0; i < library->num_albums; i++) {
        struct Album album = library->albums[i];
        printf("%s - %s (%d):\n", album.artist, album.name, album.year);
        for (int j = 0; j < album.num_songs; j++) {
            struct Song song = album.songs[j];
            printf("\t%s - %s (%d:%d)\n", song.artist, song.name, song.duration / 60, song.duration % 60);
        }
        printf("\n");
    }
}

int main() {
    struct Library library;
    library.num_albums = 0;
    add_album(&library, "The Dark Side of the Moon", "Pink Floyd", 1973);
    add_album(&library, "Led Zeppelin IV", "Led Zeppelin", 1971);
    add_album(&library, "Rumours", "Fleetwood Mac", 1977);
    add_song(&library, "The Dark Side of the Moon", "Speak to Me", "Pink Floyd", 67);
    add_song(&library, "The Dark Side of the Moon", "Breathe", "Pink Floyd", 163);
    add_song(&library, "Led Zeppelin IV", "Stairway to Heaven", "Led Zeppelin", 482);
    add_song(&library, "Rumours", "Go Your Own Way", "Fleetwood Mac", 219);
    print_library(&library);
    return 0;
}

