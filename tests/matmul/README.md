# HiPar22 - Test - Matmul


For the second experiment, the test runs a recursive implementation of the matrix multiplication. The algorithm partitions the matrices -- A, B and C -- into NxN parts and computes the matrix multiplications necessary to compute the resulting part and gathers all the results to produce the final result. 

Thus, when multiplying two 11x11 matrices partitioning them into two blocks per dimension, the algorithm will split all the matrices into 4 blocks.
```
(11,11) -> (8,8) (8,3)
           (3,8) (3,3)
```
And thus the result matrix would be calculated as
```
C (11,11) -> C_0_0 = A_0_0 * B_0_0 + A_0_1 * B_1_0            C_0_1 = A_0_0 * B_0_1 + A_0_1 * B_1_1
             (8,8) = (8,8) * (8,8) + (8,3) * (3,8)            (8,3) = (8,8) * (8,3) + (8,3) * (3,3)
             (8,8) =     (8,8)    +      (8,8)                (8,3) =     (8,3)    +      (8,3)


             C_1_0 = A_1_0 * B_0_0 + A_1_1 * B_1_0            C_1_1 = A_1_0 * B_0_1 + A_1_1 * B_1_1
             (3,8) = (3,8) * (8,8) + (3,3) * (3,8)            (3,3) = (3,8) * (8,3) + (3,3) * (3,3)
             (3,8) =     (3,8)    +      (3,8)                (3,3) =     (3,3)    +      (3,3)
```

In turn, each part multiplication is computed recursively until the matrix to multiply reaches the desired block size. At that point, the computation relies on NumPy to compute the multiplication result.

The Nested version detects as a task every multiplication of two matrices regardless of their size -- including the starting one -- and every addition. Conversely, the Flat version creates a task only for matrix multiplications of two blocks and additions. Therefore, the Flat version generates all the tasks from the main code, whereas the Nested version creates the task graph as a hierarchy.

# APPLICATION SOURCE CODE
The `application` folder contains the source files for the two different implementations of the application.

The `recursive_matmul_flat.py` file is the source code for the `FLAT` version of the application. In this version, the main code of the application executes all the recursivity layers generating all the tasks. In this case, only the multiplication of block-sized matrices and the recursive addition of the results are selected as tasks.

Conversely, the `NESTED` version, whose code can be found in the `recursive_matmul_paths.py` file, creates a workflow hierarchy leveraging on the recursivity of the application by converting into tasks EVERY multiplication of matrices and the addition of their results.

# Test Execution
To launch the application, users can directly call the `launch` script within the test folder passing in the indicated parameters for the version which are:
- NUM_MATMULS: number of matrix multiplications launched subsequently
- MATRIX_SIZE: number of elements per dimension of the matrices (the paper uses 56,536)
- BLOCK_SIZE: number of elements per dimension of the blocks composing the matrices (the paper uses 4,096)
- RECURSIVE_SPLIT: numer of parts per dimension into which the matrices are split on each recursive step (the paper uses 2, 4, and 16)

```bash
> launch <FLAT|NESTED> [NUM_MODELS [MATRIX_SIZE [BLOCK_SIZE [RECURSIVE_SPLIT]]]]
```

Otherwise, the user can use the `launch_test` script in the root folder of the repository or the container passing in `random_forest` as the first parameter . Both ways end up calling the launch script removing the application name.
```bash
> launch_test random_forest <FLAT|NESTED> [NUM_MODELS [MATRIX_SIZE [BLOCK_SIZE [RECURSIVE_SPLIT]]]]
> docker run --rm compss/hipar22:latest random_forest <FLAT|NESTED> [NUM_MODELS [MATRIX_SIZE [BLOCK_SIZE [RECURSIVE_SPLIT]]]]
```

The script `enqueue` can be used to submit the execution onto the queue system of a supercomputer. Execution times can be retrieved with the `get_times` script which returns a list with an entry for each execution detailing the execution id, the number of nodes used, the number of estimators and the training time in ms for each execution.
```bash
> enqueue <num_nodes> <FLAT|NESTED> [NUM_MODELS [MATRIX_SIZE [BLOCK_SIZE [RECURSIVE_SPLIT]]]]
> get_times
```