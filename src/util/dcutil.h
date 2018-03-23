
#ifndef UTIL_DIGITAL_CONTROLLERS_DCUTIL_H_
#define UTIL_DIGITAL_CONTROLLERS_DCUTIL_H_

/** @file dcutil.h
 *
 *  Class for help in digital controllers verification
 *
 *  @author Hussama Ibrahim
 */
class dcutil
{
public:
  dcutil() = default;
  virtual ~dcutil() = default;

  /**
	 * Method to generate delta binomial.
	 *
	 * @param grau is the level of binomial
	 * @param delta is the delta value
	 * @param out is the binomial in array (output).
	 */
  void delta_binomial_generation(int grau, float delta, float out[]);

  /**
	 * Method that generate delta coefficients for
	 * denominators.
	 *
	 * @param vetor is the denominator array
	 * @param out is the delta coefficients generated array (output)
	 * @param n is the size of arrays vetor, out (same)
	 * @param delta is the delta value
	 */
  void
  generate_delta_coefficients(float vetor[], float out[], int n, float delta);

  /**
	 * Method that generate delta coefficients
	 * for numerator. It is similar as above method
	 * but for de numerator is necessary divide the
	 * generated coefficients by delta.
	 *
	 * @param vetor_b is the numerator array
	 * @param out is the delta coefficients generated array (output)
	 * @param n is the size of arrays vetor_b, out (same)
	 * @param delta is the delta value
	 */
  void generate_delta_coefficients_b(
    float vetor_b[],
    float out[],
    int n,
    float delta);

  /**
	 * Method that revert an array position.
	 * Ex: { 1, 2, 3 } -> { 3, 2, 1 }
	 *
	 * @param v is a array
	 * @param out is a output array (inverted)
	 * @param n is size of array v
	 */
  void revert_array(float v[], float out[], int n);

  /**
	 * Method that init the array with zeroes.
	 *
	 * @param v is the array
	 * @param n is the array size
	 */
  void init_array(float v[], int n);

  /**
	 * Method that check a stability from a
	 * digital controller denominator
	 *
	 * @param a is the denominator of digital controller
	 * @param n is the size of a
	 */
  int check_stability(float a[], int n);

  /**
	 * Simple method to generate a pow of a number
	 *
	 * @param a is the base
	 * @param b is de expoent.
	 */
  double pow(double a, double b);

  /**
	 * Simple method to generate a binomial coefficient.
	 *
	 * @param n
	 * @param p
	 */
  int binomial_coefficient(int n, int p);

  /**
	 * Simple method to generate a fatorial of a number.
	 *
	 * @param n is the number
	 */
  int fatorial(int n);
};

#endif /* UTIL_DIGITAL_CONTROLLERS_DCUTIL_H_ */
