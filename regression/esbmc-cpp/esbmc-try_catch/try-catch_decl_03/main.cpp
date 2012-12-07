void myfunction () throw (char) {
  try {
    throw 5.0;
  } catch (double) {
    throw 'x';
  }
}

int main (void) {
  try {
    try {
      myfunction();
    } catch(...) {
      throw 'x';
    }
  }
  catch (int) { return 3; }
  catch (char) { return 2; }
  catch (...) { return 1; }
  return 0;
}
