#include <util/usr_utils.h>
#include <vector>

namespace
{
/**
 * Check if a string is already in USR format.
 * USR format: c:[@][@N@ns][@S@Class]@F@func[#]
 */
bool is_usr_format(std::string_view str)
{
  // All USRs start with "c:" and contain "@F@" (function marker)
  return str.length() >= 3 && str.substr(0, 2) == "c:" &&
         str.find("@F@") != std::string_view::npos;
}

/**
 * Ensure USR string ends with '#'
 */
std::string ensure_usr_terminator(std::string_view usr)
{
  std::string result(usr);
  if (result.back() != '#')
    result += '#';
  return result;
}
} // namespace

std::string user_name_to_usr(std::string_view user_name)
{
  // If already in USR format, just ensure trailing #
  if (is_usr_format(user_name))
    return ensure_usr_terminator(user_name);

  // Split by '@' to parse scope components
  std::vector<std::string> parts;
  std::string::size_type pos = 0;
  while (pos < user_name.length())
  {
    std::string::size_type next = user_name.find('@', pos);
    if (next == std::string::npos)
    {
      parts.push_back(std::string(user_name.substr(pos)));
      break;
    }
    parts.push_back(std::string(user_name.substr(pos, next - pos)));
    pos = next + 1;
  }

  if (parts.empty())
    return "";

  // Build USR: c:[file][@N@ns][@S@Class]@F@func#
  std::string usr = "c:";
  std::string func_name;
  size_t i = 0;

  // Check if first part is a file (contains . or /)
  if (
    parts[0].find('.') != std::string::npos ||
    parts[0].find('/') != std::string::npos)
  {
    usr += parts[0];
    i = 1;
  }

  // Process scope prefixes (N@ and S@)
  while (i < parts.size())
  {
    if (i + 1 < parts.size() && (parts[i] == "N" || parts[i] == "S"))
    {
      // Scope prefix: @N@name or @S@name
      usr += "@" + parts[i] + "@" + parts[i + 1];
      i += 2;
    }
    else
    {
      // Last part is the function name
      func_name = parts[i];
      break;
    }
  }

  // Add function marker and name
  if (!func_name.empty())
  {
    usr += "@F@" + func_name + "#";
    return usr;
  }

  return "";
}

std::string usr_to_user_name(std::string_view usr_name)
{
  // Check if it's in USR format (starts with c:)
  if (usr_name.length() < 3 || usr_name.substr(0, 2) != "c:")
    return std::string(usr_name); // Not in USR format, return as-is

  // Always strip "c:" and handle optional "@" or "file@" prefix uniformly
  std::string usr(usr_name.substr(2)); // Strip "c:"
  std::string user_name;
  size_t pos = 0;

  if (!usr.empty() && usr[0] == '@')
  {
    // Regular format: starts with @ (e.g., c:@F@func#)
    // Strip the @ and continue parsing
    usr = usr.substr(1);
  }
  else
  {
    // File-scoped format: starts with filename (e.g., c:file@F@func#)
    size_t first_at = usr.find('@');
    if (first_at == std::string::npos)
      return std::string(usr_name); // No @, invalid format

    // Extract filename prefix
    user_name = usr.substr(0, first_at) + "@";
    pos = first_at + 1;
  }

  // Parse USR components: [@N@ns][@S@Class]@F@func#

  // Process scope markers
  while (pos < usr.length())
  {
    if (pos + 2 < usr.length() && usr[pos] == 'N' && usr[pos + 1] == '@')
    {
      // Namespace: N@name@
      size_t end = usr.find('@', pos + 2);
      if (end != std::string::npos)
      {
        user_name += "N@" + usr.substr(pos + 2, end - pos - 2) + "@";
        pos = end + 1;
      }
      else
        break;
    }
    else if (pos + 2 < usr.length() && usr[pos] == 'S' && usr[pos + 1] == '@')
    {
      // Class/Struct: S@name@
      size_t end = usr.find('@', pos + 2);
      if (end != std::string::npos)
      {
        user_name += "S@" + usr.substr(pos + 2, end - pos - 2) + "@";
        pos = end + 1;
      }
      else
        break;
    }
    else if (pos + 2 < usr.length() && usr[pos] == 'F' && usr[pos + 1] == '@')
    {
      // Function: F@name#
      size_t end = usr.find('#', pos + 2);
      if (end != std::string::npos)
        user_name += usr.substr(pos + 2, end - pos - 2);
      else
        user_name += usr.substr(pos + 2); // No trailing #
      break;
    }
    else
      pos++;
  }

  return user_name.empty() ? std::string(usr_name) : user_name;
}
