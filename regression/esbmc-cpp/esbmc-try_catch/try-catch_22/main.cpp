void myfunction () throw (char) {
  try {
    throw 5.0f;
  } catch (float) {
    throw 'x';
  }
}

int main (void) {
  try {
    myfunction();
  }
  catch (int) { return 3; }
  catch (char) { return 2; }
  catch (...) { return 1; }
  return 0;
}
