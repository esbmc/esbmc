#pragma once

#include <algorithm>
#include <string>

// Paths of the Python operational models. The parser records each module's
// path as the operating system spells it, '\'-separated on Windows, so every
// check compares the '/'-separated form.
namespace model_path
{
inline std::string normalised(std::string path)
{
  std::replace(path.begin(), path.end(), '\\', '/');
  return path;
}

inline bool in_models(const std::string &path)
{
  return normalised(path).find("/models/") != std::string::npos;
}

// Whether `path` is the model file `file`, e.g. "numpy.py".
inline bool is_model(const std::string &path, const std::string &file)
{
  const std::string suffix = "/models/" + file;
  const std::string p = normalised(path);
  return p.size() >= suffix.size() &&
         p.compare(p.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The NumPy model or one of its numpy.linalg modules.
inline bool is_numpy_model(const std::string &path)
{
  return is_model(path, "numpy.py") ||
         normalised(path).find("/numpy/linalg") != std::string::npos;
}
} // namespace model_path
