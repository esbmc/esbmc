#pragma once

/**
 * @brief Registers every file bundled into the binary with file_operations.
 *
 * Must run before anything reads a bundled path. Registration only records
 * pointers into .rodata, so doing it all up front costs nothing measurable and
 * removes the need for each frontend to track whether it has extracted yet.
 */
void register_bundled_files();
