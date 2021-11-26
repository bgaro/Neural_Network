/* Function to perform backpropagation of error.
 *
 * Arguments:
 *   - network: pointer to network
 *   - example: pointer to example
 *   - eta: learning rate
 *   - alpha: momentum
 *   - lambda: weight decay
 *   - bp_type: type of backpropagation
 *   - bp_delta: pointer to error delta
 *   - bp_weight: pointer to weight delta
 *   - bp_bias: pointer to bias delta
 *
 * Return:
 *   - error: error
 */
