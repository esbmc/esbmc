//GEMINI-pro DATASET v1.0 Category: Disk space analyzer ; Style: lively
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

// Function to get file size in bytes
long long get_file_size(const char *path)
{
  struct stat statbuf;
  if (stat(path, &statbuf) == -1)
  {
    perror("stat");
    return -1;
  }
  return statbuf.st_size;
}

// Function to print file size in a human-readable format
void print_file_size(long long size)
{
  if (size < 1024)
  {
    printf("%lld bytes\n", size);
  }
  else if (size < 1048576)
  {
    printf("%.2f KB\n", (double)size / 1024);
  }
  else if (size < 1073741824)
  {
    printf("%.2f MB\n", (double)size / 1048576);
  }
  else
  {
    printf("%.2f GB\n", (double)size / 1073741824);
  }
}

// Function to analyze disk space usage
void analyze_disk_space(const char *path)
{
  DIR *dir;
  struct dirent *entry;
  long long total_size = 0;

  // Open the directory
  if ((dir = opendir(path)) == NULL)
  {
    perror("opendir");
    return;
  }

  // Iterate over the directory entries
  while ((entry = readdir(dir)) != NULL)
  {
    // Skip current and parent directories
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
    {
      continue;
    }

    // Get the file size
    char full_path[strlen(path) + strlen(entry->d_name) + 2];
    sprintf(full_path, "%s/%s", path, entry->d_name);
    long long file_size = get_file_size(full_path);
    if (file_size == -1)
    {
      continue;
    }

    // Add the file size to the total
    total_size += file_size;
  }

  // Close the directory
  closedir(dir);

  // Print the total size
  printf("Total disk space used: ");
  print_file_size(total_size);
}

int main(int argc, char *argv[])
{
  // Check if a path was specified
  if (argc != 2)
  {
    printf("Usage: %s <path>\n", argv[0]);
    return 1;
  }

  // Analyze the disk space usage
  analyze_disk_space(argv[1]);

  return 0;
}
