#include <stdint.h>

static int64_t
get_value(int64_t *base, int64_t rows, int64_t cols, int64_t row, int64_t col)
{
  int64_t effective_row = rows == 1 ? 0 : row;
  int64_t effective_col = cols == 1 ? 0 : col;
  return *(base + effective_row * cols + effective_col);
}

static double get_value_double(
  double *base,
  int64_t rows,
  int64_t cols,
  int64_t row,
  int64_t col)
{
  int64_t effective_row = rows == 1 ? 0 : row;
  int64_t effective_col = cols == 1 ? 0 : col;
  return *(base + effective_row * cols + effective_col);
}

void add(
  int64_t *A,
  int64_t *B,
  int64_t *C,
  int64_t a_rows,
  int64_t a_cols,
  int64_t b_rows,
  int64_t b_cols)
{
  int64_t rows = a_rows > b_rows ? a_rows : b_rows;
  int64_t cols = a_cols > b_cols ? a_cols : b_cols;

  int64_t i = 0;
  while (i < rows)
  {
    int64_t j = 0;
    while (j < cols)
    {
      *(C + i * cols + j) =
        get_value(A, a_rows, a_cols, i, j) + get_value(B, b_rows, b_cols, i, j);
      j++;
    }
    i++;
  }
}

void add_double(
  double *A,
  double *B,
  double *C,
  int64_t a_rows,
  int64_t a_cols,
  int64_t b_rows,
  int64_t b_cols)
{
  int64_t rows = a_rows > b_rows ? a_rows : b_rows;
  int64_t cols = a_cols > b_cols ? a_cols : b_cols;

  int64_t i = 0;
  while (i < rows)
  {
    int64_t j = 0;
    while (j < cols)
    {
      *(C + i * cols + j) = get_value_double(A, a_rows, a_cols, i, j) +
                            get_value_double(B, b_rows, b_cols, i, j);
      j++;
    }
    i++;
  }
}

void subtract(
  int64_t *A,
  int64_t *B,
  int64_t *C,
  int64_t a_rows,
  int64_t a_cols,
  int64_t b_rows,
  int64_t b_cols)
{
  int64_t rows = a_rows > b_rows ? a_rows : b_rows;
  int64_t cols = a_cols > b_cols ? a_cols : b_cols;

  int64_t i = 0;
  while (i < rows)
  {
    int64_t j = 0;
    while (j < cols)
    {
      *(C + i * cols + j) =
        get_value(A, a_rows, a_cols, i, j) - get_value(B, b_rows, b_cols, i, j);
      j++;
    }
    i++;
  }
}

void subtract_double(
  double *A,
  double *B,
  double *C,
  int64_t a_rows,
  int64_t a_cols,
  int64_t b_rows,
  int64_t b_cols)
{
  int64_t rows = a_rows > b_rows ? a_rows : b_rows;
  int64_t cols = a_cols > b_cols ? a_cols : b_cols;

  int64_t i = 0;
  while (i < rows)
  {
    int64_t j = 0;
    while (j < cols)
    {
      *(C + i * cols + j) = get_value_double(A, a_rows, a_cols, i, j) -
                            get_value_double(B, b_rows, b_cols, i, j);
      j++;
    }
    i++;
  }
}

void multiply(
  int64_t *A,
  int64_t *B,
  int64_t *C,
  int64_t a_rows,
  int64_t a_cols,
  int64_t b_rows,
  int64_t b_cols)
{
  int64_t rows = a_rows > b_rows ? a_rows : b_rows;
  int64_t cols = a_cols > b_cols ? a_cols : b_cols;

  int64_t i = 0;
  while (i < rows)
  {
    int64_t j = 0;
    while (j < cols)
    {
      *(C + i * cols + j) =
        get_value(A, a_rows, a_cols, i, j) * get_value(B, b_rows, b_cols, i, j);
      j++;
    }
    i++;
  }
}

void multiply_double(
  double *A,
  double *B,
  double *C,
  int64_t a_rows,
  int64_t a_cols,
  int64_t b_rows,
  int64_t b_cols)
{
  int64_t rows = a_rows > b_rows ? a_rows : b_rows;
  int64_t cols = a_cols > b_cols ? a_cols : b_cols;

  int64_t i = 0;
  while (i < rows)
  {
    int64_t j = 0;
    while (j < cols)
    {
      *(C + i * cols + j) = get_value_double(A, a_rows, a_cols, i, j) *
                            get_value_double(B, b_rows, b_cols, i, j);
      j++;
    }
    i++;
  }
}

void divide(
  int64_t *A,
  int64_t *B,
  int64_t *C,
  int64_t a_rows,
  int64_t a_cols,
  int64_t b_rows,
  int64_t b_cols)
{
  int64_t rows = a_rows > b_rows ? a_rows : b_rows;
  int64_t cols = a_cols > b_cols ? a_cols : b_cols;

  int64_t i = 0;
  while (i < rows)
  {
    int64_t j = 0;
    while (j < cols)
    {
      *(C + i * cols + j) =
        get_value(A, a_rows, a_cols, i, j) / get_value(B, b_rows, b_cols, i, j);
      j++;
    }
    i++;
  }
}

void divide_double(
  double *A,
  double *B,
  double *C,
  int64_t a_rows,
  int64_t a_cols,
  int64_t b_rows,
  int64_t b_cols)
{
  int64_t rows = a_rows > b_rows ? a_rows : b_rows;
  int64_t cols = a_cols > b_cols ? a_cols : b_cols;

  int64_t i = 0;
  while (i < rows)
  {
    int64_t j = 0;
    while (j < cols)
    {
      *(C + i * cols + j) = get_value_double(A, a_rows, a_cols, i, j) /
                            get_value_double(B, b_rows, b_cols, i, j);
      j++;
    }
    i++;
  }
}
