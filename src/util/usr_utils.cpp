#include <util/usr_utils.h>
#include <vector>

std::string user_name_to_usr(const std::string &user_name)
{
  // If already in internal USR format, ensure trailing #
  if (user_name.length() >= 3 && user_name.substr(0, 3) == "c:@")
  {
    std::string usr = user_name;
    if (usr.back() != '#')
      usr += '#';
    return usr;
  }

  // Split by '@' to parse scope components
  std::vector<std::string> parts;
  std::string::size_type pos = 0;
  while (pos < user_name.length())
  {
    std::string::size_type next = user_name.find('@', pos);
    if (next == std::string::npos)
    {
      parts.push_back(user_name.substr(pos));
      break;
    }
    parts.push_back(user_name.substr(pos, next - pos));
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

std::string usr_to_user_name(const std::string &usr_name)
{
  // Strip "c:@" prefix if present
  if (usr_name.length() < 3 || usr_name.substr(0, 3) != "c:@")
    return usr_name; // Not in USR format, return as-is

  std::string usr = usr_name.substr(3); // Strip "c:@" prefix
  std::string user_name;

  // Parse USR components: [file][@N@ns][@S@Class]@F@func#
  size_t pos = 0;

  // Check for file prefix (before first @ or before @F@)
  size_t first_at = usr.find('@');
  if (first_at != std::string::npos)
  {
    std::string potential_file = usr.substr(0, first_at);
    if (
      potential_file.find('.') != std::string::npos ||
      potential_file.find('/') != std::string::npos)
    {
      user_name = potential_file + "@";
      pos = first_at + 1;
    }
  }

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

  return user_name.empty() ? usr_name : user_name;
}
