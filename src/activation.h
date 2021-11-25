/**
 * @brief unit step activation functions
 * @param x input : inner potential
 * @return 0 if x<0, 1 if x>=0
 **/
float unit_step(float x);

/**
 * @brief ReLU activation functions
 * @param x input : inner potential
 * @return 0 if x<0, x if x>=0
 **/
float reLU(float x);