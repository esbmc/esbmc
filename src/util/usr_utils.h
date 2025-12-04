#ifndef CPROVER_UTIL_USR_UTILS_H
#define CPROVER_UTIL_USR_UTILS_H

#include <string>

/**
 * Convert user-friendly function name to Clang USR format.
 *
 * Supported formats:
 *  - Function: func → c:@F@func#
 *  - Namespace: N@ns@func → c:@N@ns@F@func#
 *  - Class: S@Class@method → c:@S@Class@F@method#
 *  - Static: file.c@func → c:file.c@F@func#
 *  - Composite: file.c@N@ns@S@Class@method → c:file.c@N@ns@S@Class@F@method#
 *  - Passthrough: c:@F@func# → c:@F@func# (already in USR format)
 *
 * @param user_name User-friendly function name
 * @return Clang USR format with trailing #
 */
std::string user_name_to_usr(const std::string &user_name);

/**
 * Convert Clang USR format to user-friendly function name.
 *
 * Inverse of user_name_to_usr. Strips c:@ prefix and converts scope markers
 * (N@, S@, F@) to user-friendly format.
 *
 * @param usr_name Clang USR format (e.g., c:@F@func# or c:@N@ns@S@Class@F@method#)
 * @return User-friendly name (e.g., func or N@ns@S@Class@method)
 */
std::string usr_to_user_name(const std::string &usr_name);

#endif
