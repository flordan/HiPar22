from ast import Str
from sys import warnoptions
from pycompss.api.task import task
from pycompss.api.api import compss_barrier
from pycompss.api.parameter import *
import time
import numpy as np
import math
import random
import os
import random
import string

def create_with_random(block_name, size_y, size_x):
    block = np.random.rand(size_y, size_x)
    np.save(block_name, block)

def create_matrix(matrix_name, init_x, init_y, max_x, max_y, block_x, block_y, split_x, split_y, fill_function=None):
    x_to_cover = max_x - init_x
    y_to_cover = max_y - init_y
    num_x_blocks=math.ceil(x_to_cover/block_x)
    num_y_blocks=math.ceil(y_to_cover/block_y)
    blocks_per_x_partition = math.ceil(num_x_blocks / split_x)
    x_stride=blocks_per_x_partition*block_x
    blocks_per_y_partition = math.ceil(num_y_blocks / split_y)
    y_stride=blocks_per_y_partition*block_y

    result=[]
    x = init_x
    while x < max_x:
        row=[]
        next_x = x + x_stride
        if next_x > max_x:
            next_x = max_x
        y = init_y
        while y < max_y:
            next_y = y + y_stride
            if next_y > max_y:
                next_y = max_y

            if (next_x > x+block_x) or (next_y >y +block_y):
                row.append(create_matrix(matrix_name, x, y, next_x, next_y, block_x, block_y, split_x, split_y, fill_function))
            else: 
                block_name = matrix_name + "_" + str(x) + "_" + str(y) + ".npy"
                # block_name = os.path.abspath(block_name)
                if fill_function is not None:
                    fill_function(block_name, next_x - x, next_y -y)
                row.append(block_name)

            y = next_y
        x = next_x
        result.append(row)
    return result

def replicate_structure_in_array(elem, tmp_files_path):
    if isinstance(elem, str):
        block_name=os.path.join(tmp_files_path, "tmp."+str(random.getrandbits(64))+".npy" )
        return block_name

    else:
        arr = [0] * len(elem)
        for i in range(len(elem)):
            arr[i] = replicate_structure_in_array(elem[i], tmp_files_path)
        return arr

@task(
    addends={Type: COLLECTION_IN},
    result={Type: COLLECTION_IN},
    dependency_tokens={Type: COLLECTION_IN}
)
def addition(addends, result, dependency_tokens):
    if len(result) == 1 and len(result[0]) == 1:
        addend_blocks=[]
        for addend in addends:
            addend_blocks.append(addend[0][0])
        block_addition(addend_blocks, result[0][0])
                
    else:
        for i in range(len(result)):
            for j in range(len(result[0])):
                addend_subblocks=[]
                for addend in addends:
                    addend_subblocks.append(addend[i][j])
                    addition_aux(addend_subblocks, result[i][j])
    
    recursive_delete(addends)
    


def addition_aux(addends, result):
    if isinstance(result, str):
        block_addition(addends, result)
    else:
        for i in range(len(result)):
            for j in range(len(result[0])):
                addend_subblocks=[]
                for addend in addends:
                    addend_subblocks.append(addend[i][j])
                    addition_aux(addend_subblocks, result[i][j])


def block_addition(addends, result):
    accum=None
    for addend in addends:
        block = np.load(addend)
        if accum is None:
            accum = block
        else:
            h,w = block.shape
            accum[0:h,0:w] += block[0:h,0:w]
    np.save(result, accum)


@task(
    matA={Type: COLLECTION_IN},
    matB={Type: COLLECTION_IN},
    matC={Type: COLLECTION_IN}
)
def matmul(matA, matB, matC, tmp_files_path):
    if len(matC) == 1 and len(matC[0]) == 1:
        res = None
        for i in range(len(matA)):
            A = np.load(matA[i][0])
            B = np.load(matB[0][i])
            partial = np.matmul(A,B)
            if res is None:
                res = partial
            else:
                h,w = partial.shape
                res[0:h,0:w] += partial[0:h,0:w]
        np.save(matC[0][0], res)

    elif len(matA) == 1 and len(matA[0]) == 1:
        for j in range(len(matB[0])):
            matmul(matA, as_collection(matB[0][j]), as_collection(matC[0][j]), tmp_files_path)
    elif len(matB) == 1 and len(matB[0]) == 1:
        for i in range(len(matA)):
            matmul(as_collection(matA[i][0]), matB, as_collection(matC[i][0]), tmp_files_path)
    else:
        for i in range(len(matC)):
            for j in range(len(matC[0])):
                if len(matB) == 1:
                    matmul(as_collection (matA[i][0]), as_collection(matB[0][j]), as_collection(matC[i][j]), tmp_files_path)
                else:
                    partials = []
                    dependency_tokens = []
                    for k in range(len(matB)):
                        partial = replicate_structure_in_array(matC[i][j], tmp_files_path)
                        dependency_token = matmul(as_collection (matA[i][k]), as_collection(matB[k][j]), as_collection(partial), tmp_files_path)
                        partials.append(as_collection(partial))
                        dependency_tokens.append(dependency_token)
                    addition(partials, as_collection(matC[i][j]), dependency_tokens)
                    # compss_delete_object(dependency_tokens)

    return object()

def as_collection(mat):
    if isinstance(mat, str):
        return [[mat]]
    else:
        return mat

def recursive_delete(mat):
    if isinstance(mat, str):
        os.remove(mat)
    else:
        for elem in mat:
            recursive_delete(elem)
            
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def recursiveAppend(elem):
    if isinstance(elem, str):
        res = np.load(elem)
        return res
    else:
        acumulator = None
        for i in range(len(elem)):
            acumulator_x = None
            for j in range(len(elem[0])):
                if acumulator_x is None:
                    acumulator_x = recursiveAppend(elem[i][j])
                else:
                    adding_element = recursiveAppend(elem[i][j])
                    acumulator_x = np.append(acumulator_x, adding_element, axis=1)
            if acumulator is None:
                acumulator = acumulator_x
            else:
                acumulator = np.append(acumulator, acumulator_x, axis=0)
        return acumulator

def test_correct_result(A, B, C):
    A = recursiveAppend(A)
    B = recursiveAppend(B)
    C = recursiveAppend(C)
    np_result = np.matmul(A, B)
    if not np.allclose(C, np_result):
        print("Multiplication result is not correct", flush=True)
    else:
        print("Multiplication result is OK", flush=True)

def test_matmul(x, y, z, b_x, b_y, split_x, split_y, num_matmuls, tmp_files_path):
    print("----------------------------------------------------------------")
    A_size = "(" + str(x) + "," + str(y) + ")"
    B_size = "(" + str(y) + "," + str(z) + ")"
    C_size = "(" + str(x) + "," + str(z) + ")"
    print("Multiplying " + A_size + " * " + B_size+ " = " + C_size, flush=True)
    A_block_size = "(" + str(b_x) + "," + str(b_y) + ")"
    B_block_size = "(" + str(b_y) + "," + str(b_x) + ")"
    C_block_size = "(" + str(b_x) + "," + str(b_x) + ")"
    print("Divided into blocks of " + A_block_size + " * " + B_block_size+ " = " + C_block_size, flush=True)
   
    initialization_matrix_t0 = time.time()
    tmp_files_path = os.path.join(tmp_files_path, get_random_string(12))
    os.mkdir(tmp_files_path)
    A = create_matrix(os.path.join(tmp_files_path,"A"), 0, 0, x, y, b_x, b_y, split_x, split_y, create_with_random)
    B = create_matrix(os.path.join(tmp_files_path,"B"), 0, 0, y, z, b_y, b_x, split_y, split_x, create_with_random)
    compss_barrier()
    initialization_matrix_t1 = time.time()
    print("Time spent on matrices A and B creation: ", int((initialization_matrix_t1 - initialization_matrix_t0) *1000), flush=True)
    print("", flush=True)

    for i in range(0, num_matmuls + 1):
        print("Allocating result matrix", flush=True)
        initialization_matrix_t0 = time.time()
        C = create_matrix(os.path.join(tmp_files_path,"C"), 0, 0, x, z, b_x, b_x, split_x, split_x, None)
        compss_barrier()
        initialization_matrix_t1 = time.time()
        print("Time spent on result matrix allocation: ", int((initialization_matrix_t1 - initialization_matrix_t0) *1000), flush=True)

        matmul_t0 = time.time()
        matmul(A, B, C, tmp_files_path)
        compss_barrier()
        matmul_t1 = time.time()
        print("Time spent multiplying: ", int((matmul_t1 - matmul_t0) * 1000 ), flush=True )
        # test_correct_result(A, B, C)
        recursive_delete(C)
    recursive_delete(B)
    recursive_delete(A)


@task()
def main(num_matmuls_str="5", mat_size_str="4", block_size_str="2", partitions_str="2", tmp_files_path=os.path.abspath(os.getcwd())):
    num_matmuls = int(num_matmuls_str)
    mat_size = int(mat_size_str)
    block_Size = int(block_size_str)
    partitions = int(partitions_str)

    x = mat_size
    y = mat_size
    z = mat_size

    b_x = block_Size
    b_y = block_Size

    split_x = partitions
    split_y = partitions

    os.makedirs(tmp_files_path, exist_ok=True)

    test_matmul(x, y, z, b_x, b_y, split_x, split_y, num_matmuls, tmp_files_path)

if __name__ == "__main__":
    main("1", "8", "2", "2")



