import numpy as np

def swap_rows(M, row_index_1, row_index_2):

    M = M.copy()
    # Swap indexes
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M

def get_index_first_non_zero_value_from_column(M, column, starting_row):

    # Get the column array starting from the specified row
    column_array = M[starting_row:,column]
    for i, val in enumerate(column_array):
        if not np.isclose(val, 0, atol = 1e-5):
            index = i + starting_row
            return index
    return -1

def get_index_first_non_zero_value_from_row(M, row, augmented = False):
    # Create a copy to avoid modifying the original matrix
    M = M.copy()


    # If it is an augmented matrix, then ignore the constant values
    if augmented == True:
        # Isolating the coefficient matrix (removing the constant terms)
        M = M[:,:-1]
        
    # Get the desired row
    row_array = M[row]
    for i, val in enumerate(row_array):
        # If finds a non zero value, returns the index. Otherwise returns -1.
        if not np.isclose(val, 0, atol = 1e-5):
            return i
    return -1


def augmented_matrix(A, B):
    augmented_M = np.hstack((A,B))
    return augmented_M



def row_echelon_form(A, B):
    
    # Before any computation, check if matrix A (coefficient matrix) has non-zero determinant. 
    # It will be used the numpy sub library np.linalg to compute it.

    det_A = np.linalg.det(A)

    # Returns "Singular system" if determinant is zero
    if np.isclose(det_A, 0) == True:
        return 'Singular system'

    # Make copies of the input matrices to avoid modifying the originals
    A = A.copy()
    B = B.copy()


    # Convert matrices to float to prevent integer division
    A = A.astype('float64')
    B = B.astype('float64')

    # Number of rows in the coefficient matrix
    num_rows = len(A) 

    ### START CODE HERE ###

    # Transform matrices A and B into the augmented matrix M
    M = augmented_matrix(A,B)
    
    # Iterate over the rows.
    for row in range(num_rows):

        # The first pivot candidate is always in the main diagonal, let's get it. 
        # Remember that the diagonal elements in a matrix has the same index for row and column. 
        # You may access a matrix value by typing M[row, column]. In this case, column = None
        pivot_candidate = M[row, row]

        # If pivot_candidate is zero, it cannot be a pivot for this row. 
        # So the first step you need to take is to look at the rows below it to check if there is a non-zero element in the same column.
        # The usage of np.isclose is a good practice when comparing two floats.
        if np.isclose(pivot_candidate, 0) == True: 
            # Get the index of the first non-zero value below the pivot_candidate. 
            first_non_zero_value_below_pivot_candidate = get_index_first_non_zero_value_from_column(M, row, row) 

            # Swap rows
            M = swap_rows(M, row, first_non_zero_value_below_pivot_candidate) 

            # Get the pivot, which is in the main diagonal now 
            pivot = M[row,row] 
        
        # If pivot_candidate is already non-zero, then it is the pivot for this row
        else:
            pivot = pivot_candidate 
        
        # Now you are ready to apply the row reduction in every row below the current
            
        # Divide the current row by the pivot, so the new pivot will be 1. You may use the formula current_row -> 1/pivot * current_row
        # Where current_row can be accessed using M[row].
        M[row] = 1/pivot * M[row]

        # Perform row reduction for rows below the current row
        for j in range(row + 1, num_rows): 
            # Get the value in the row that is below the pivot value. 
            # Remember that, since you are dealing only with non-singular matrices, the pivot is in the main diagonal.
            # Therefore, the values in row j that are below the pivot, must have column index the same index as the column index for the pivot.
            value_below_pivot = M[j,row]
            
            # Perform row reduction using the formula:
            # row_to_reduce -> row_toj_reduce - value_below_pivot * pivot_row
            M[j] = M[j] - value_below_pivot*M[row]
            
    ### END CODE HERE ###
    return M



def back_substitution(M):
    
    # Make a copy of the input matrix to avoid modifying the original
    M = M.copy()

    # Get the number of rows (and columns) in the matrix of coefficients
    num_rows = M.shape[0]

    ### START CODE HERE ####
    
    # Iterate from bottom to top
    for row in reversed(range(num_rows)): 
        substitution_row = M[row]

        # Get the index of the first non-zero element in the substitution row. Remember to pass the correct value to the argument augmented.
        index = get_index_first_non_zero_value_from_row(M, row, augmented=True)

        # Iterate over the rows above the substitution_row
        for j in range(row): 

            # Get the row to be reduced. The indexing here is similar as above, with the row variable replaced by the j variable.
            row_to_reduce = M[j]

            # Get the value of the element at the found index in the row to reduce
            value = row_to_reduce[index]
            
            # Perform the back substitution step using the formula row_to_reduce -> row_to_reduce - value * substitution_row
            row_to_reduce = row_to_reduce - value*substitution_row

            # Replace the updated row in the matrix, be careful with indexing!
            M[j,:] = row_to_reduce

    ### END CODE HERE ####

     # Extract the solution from the last column
    solution = M[:,-1]
    
    return solution


A = np.array([[1,2,3], [3,4,5], [4,5,6]])
B = np.array([[1], [5], [7]])

print(augmented_matrix(A,B))

A = np.array([[1,2,3],[0,1,0], [0,0,5]])
B = np.array([[1], [2], [4]])
C = row_echelon_form(A,B)

C_final = back_substitution(C)