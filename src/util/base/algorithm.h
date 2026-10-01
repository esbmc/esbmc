#pragma once

/**
 * @brief Base interface to run an algorithm in esbmc
 */
template <typename T>
class algorithm
{
public:
  algorithm(bool sideeffect) : sideeffect(sideeffect)
  {
  }

  virtual ~algorithm() = default;

  /**
   * @brief Executes the algorithm over a T object
   *
   * @return success of the algorithm
   */
  virtual bool run(T &) = 0;

  /**
   * @brief Says wether the algorithm is a plain analysis
   * or if it also changes the structure
   *
   */
  bool has_sideeffect()
  {
    return sideeffect;
  }

protected:
  // The algorithm changes the CFG, container in some way?
  const bool sideeffect;
};
