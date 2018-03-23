
#include <cmath>
#include <cstdlib>
#include <util/dcutil.h>

void dcutil::generate_delta_coefficients(
  float vetor[],
  float out[],
  int n,
  float delta)
{
  init_array(out, n);
  float a_invertido[n];
  init_array(a_invertido, n);
  revert_array(vetor, a_invertido, n);
  float _a[n];
  init_array(_a, n);
  int i, j;
  for(i = 0; i < n; i++)
  {
    float b[n + 1];
    init_array(b, n + 1);
    delta_binomial_generation(i, delta, b);
    for(j = 0; j < i + 1; j++)
    {
      b[j] = b[j] * a_invertido[i];
      _a[j] = _a[j] + b[j];
    }
  }
  revert_array(_a, out, n);
}

void dcutil::generate_delta_coefficients_b(
  float vetor[],
  float out[],
  int n,
  float delta)
{
  init_array(out, n);
  float a_invertido[n];
  init_array(a_invertido, n);
  revert_array(vetor, a_invertido, n);
  float _a[n];
  init_array(_a, n);
  int i, j;
  for(i = 0; i < n; i++)
  {
    float b[n + 1];
    init_array(b, n + 1);
    delta_binomial_generation(i, delta, b);
    for(j = 0; j < i + 1; j++)
    {
      b[j] = b[j] * a_invertido[i];
      _a[j] = _a[j] + b[j];
    }
    _a[i] = _a[i] / delta;
  }
  revert_array(_a, out, n);
}

int dcutil::fatorial(int n)
{
  return n == 0 ? 1 : n * fatorial(n - 1);
}

int dcutil::binomial_coefficient(int n, int p)
{
  return fatorial(n) / (fatorial(p) * fatorial(n - p));
}

void dcutil::delta_binomial_generation(int grau, float delta, float out[])
{
  init_array(out, 3);
  int i;
  for(i = 0; i <= grau; i++)
  {
    out[grau - i] = binomial_coefficient(grau, i) * pow(delta, grau - i);
  }
}

void dcutil::revert_array(float v[], float out[], int n)
{
  init_array(out, n);
  int i;
  for(i = 0; i < n; i++)
  {
    out[i] = v[n - i - 1];
  }
}

void dcutil::init_array(float v[], int n)
{
  int i;
  for(i = 0; i < n; i++)
  {
    v[i] = 0;
  }
}

int dcutil::check_stability(float a[], int n)
{
  int lines = 2 * n - 1;
  int columns = n;
  float m[lines][n];
  int i, j;
  for(i = 0; i < lines; i++)
  {
    for(j = 0; j < columns; j++)
    {
      m[i][j] = 0;
    }
  }
  for(i = 0; i < lines; i++)
  {
    for(j = 0; j < columns; j++)
    {
      if(i == 0)
      {
        m[i][j] = a[j];
        continue;
      }
      if(i % 2 != 0)
      {
        int x;
        for(x = 0; x < columns; x++)
        {
          m[i][x] = m[i - 1][columns - x - 1];
        }
        columns = columns - 1;
        j = columns;
      }
      else
      {
        m[i][j] = m[i - 2][j] - (m[i - 2][columns] / m[i - 2][0]) * m[i - 1][j];
      }
    }
  }
  int first_is_positive = m[0][0] >= 0 ? 1 : 0;
  for(i = 0; i < lines; i++)
  {
    if(i % 2 == 0)
    {
      int line_is_positive = m[i][0] >= 0 ? 1 : 0;
      if(first_is_positive != line_is_positive)
      {
        return 0;
      }
      continue;
    }
  }
  return 1;
}

double dcutil::pow(double a, double b)
{
  int i;
  double acc = 1;
  for(i = 0; i < b; i++)
  {
    acc = acc * a;
  }
  return acc;
}
