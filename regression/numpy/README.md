# 🧮 ESBMC – NumPy Math Library Mapping

This repository documents the mapping between ESBMC's math library implementations and their NumPy equivalents. These mappings help in testing and verifying floating-point behavior consistently across C and Python environments.

---

## 🧮 Mathematical & Trigonometric Functions

| ESBMC File | NumPy Equivalent       | Category      |
|------------|------------------------|---------------|
| `acos.c`   | `np.arccos`, `np.acos` | Inverse trig  |
| `atan.c`   | `np.arctan`, `np.atan` | Inverse trig  |
| `cos.c`    | `np.cos`               | Trig          |
| `sin.c`    | `np.sin`               | Trig          |

---

## 📐 Rounding & Remainders

| ESBMC File     | NumPy Equivalent              | Category             |
|----------------|-------------------------------|----------------------|
| `ceil.c`       | `np.ceil`                     | Rounding             |
| `floor.c`      | `np.floor`                    | Rounding             |
| `round.c`      | `np.round`, `np.around`       | Rounding             |
| `rint.c`       | `np.rint`                     | Rounding             |
| `trunc.c`      | `np.trunc`, `np.fix`          | Rounding             |
| `fmod.c`       | `np.fmod`                     | Remainder            |
| `remainder.c`  | `np.remainder`                | Remainder            |
| `remquo.c`     | `divmod` + sign logic         | Remainder + Quotient |

---

## ➕ Floating Point Properties

| ESBMC File    | NumPy Equivalent                    | Category             |
|---------------|-------------------------------------|----------------------|
| `copysign.c`  | `np.copysign`                       | Floating point ops   |
| `frexp.c`     | `np.frexp`                          | Float decomposition  |
| `modf.c`      | `np.modf`                           | Float decomposition  |
| `fpclassify.c`| `np.isnan`, `np.isinf`, `np.isfinite`| Classification       |

---

## 🔬 Comparisons, Extrema

| ESBMC File | NumPy Equivalent                    | Category             |
|------------|-------------------------------------|----------------------|
| `fmin.c`   | `np.fmin`                           | Min function         |
| `fmax.c`   | `np.fmax`                           | Max function         |
| `fdim.c`   | `np.maximum(x - y, 0)` (approx.)    | Difference           |

---

## 🔢 Exponents and Powers

| ESBMC File | NumPy Equivalent | Category     |
|------------|------------------|--------------|
| `exp.c`    | `np.exp`         | Exponential  |
| `pow.c`    | `np.power`       | Power        |

---

## 📏 Miscellaneous

| ESBMC File     | NumPy Equivalent         | Category              |
|----------------|--------------------------|-----------------------|
| `fabs.c`       | `np.fabs`, `np.absolute` | Absolute value        |
| `sqrt.c`       | `np.sqrt`                | Square root           |
| `nextafter.c`  | `np.nextafter`           | Floating-point step   |

