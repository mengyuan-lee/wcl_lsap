# -*- coding: UTF-8 -*-
import numpy as np
import time
import copy

SIMPLE, STARRED, PRIMED = 0, 1, 2



def minimize(matrix, deepcopy=True):
    if deepcopy:
        matrix = copy.deepcopy(matrix)
    n = len(matrix)

    # Step 1:
    # For each row of the matrix, find the smallest element and
    # subtract it from every element in its row. Go to Step 2.
    for row in matrix:
        m = min(row)
        if m != 0:
            row[:] = map(lambda x: x - m, row)

    mask_matrix = [[SIMPLE] * n for _ in matrix]
    row_cover = [False] * n
    col_cover = [False] * n

    # Step 2
    # Find a zero (Z) in the resulting matrix.  If there is
    # no starred zero in its row or column, star Z. Repeat for
    # each element in the matrix. Go to Step 3.
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            if value == 0 and not row_cover[r] and not col_cover[c]: #值为0，并且所在行和所在列均没有划出0
                mask_matrix[r][c] = STARRED
                row_cover[r] = True
                col_cover[c] = True

    row_cover = [False] * n
    col_cover = [False] * n

    # Step 3
    # Cover each column containing a starred zero.  If K columns
    # are covered, the starred zeros describe a complete set of
    # unique assignments. In this case, go to DONE, otherwise,
    # go to Step 4.

    match_found = False

    while not match_found:
        for i in range(n):
            col_cover[i] = any(mrow[i] == STARRED for mrow in mask_matrix)

        if all(col_cover):
            match_found = True
            continue
        else:
            # Step 4(, 6)
            zero = _cover_zeroes(matrix, mask_matrix, row_cover, col_cover)  

            # Step 5
            # Construct a series of alternating primed and starred zeros as
            # follows.  Let Z0 represent the uncovered primed zero found in
            # Step 4.  Let Z1 denote the starred zero in the column of Z0
            # (if any). Let Z2 denote the primed zero in the row of Z1
            # (there will always be one).  Continue until the series terminates
            # at a primed zero that has no starred zero in its column. Unstar
            # each starred zero of the series, star each primed zero of the
            # series, erase all primes and uncover every line in the matrix.
            # Return to Step 3.

            primes = [zero]
            stars = []
            while zero:
                zero = _find_star_in_col(mask_matrix, zero[1])
                if zero:
                    stars.append(zero)
                    zero = _find_prime_in_row(mask_matrix, zero[0])
                    primes.append(zero)

            # Erase existing stars
            for star in stars:
                mask_matrix[star[0]][star[1]] = SIMPLE

            # Star existing primes
            for prime in primes:
                mask_matrix[prime[0]][prime[1]] = STARRED

            # Erase remaining primes
            for r, row in enumerate(mask_matrix):
                for c, val in enumerate(row):
                    if val == PRIMED:
                        mask_matrix[r][c] = SIMPLE

            row_cover = [False] * n
            col_cover = [False] * n
            # end of step 5

        # end of step 3 while

    # reconstruct the solution
    #solution = []
    result = np.zeros((n))
    for r, row in enumerate(mask_matrix):
        for c, val in enumerate(row):
            if val == STARRED:
                #solution.append((r, c))
                result[r] = c
    return result
    #end of minimize


# Internal methods


def _cover_zeroes(matrix, mask_matrix, row_cover, col_cover):

    # Repeat steps 4 and 6 until we are ready to break out to step 5
    while True:
        zero = True

        # Step 4  划线
        # Find a noncovered zero and prime it.  If there is no
        # starred zero in the row containing this primed zero,
        # go to Step 5. Otherwise, cover this row and uncover
        # the column containing the starred zero. Continue in
        # this manner until there are no uncovered zeros left.
        # Save the smallest uncovered value and go to Step 6.

        while zero:
            zero = _find_noncovered_zero(matrix, row_cover, col_cover)
            if not zero: #没有找到
                break  # continue with step 6
            else:
                row = mask_matrix[zero[0]]
                row[zero[1]] = PRIMED   #将找到的0标记为prime

                try:
                    index = row.index(STARRED)
                except ValueError:
                    return zero  # continue with step 5

                row_cover[zero[0]] = True #行打钩
                col_cover[index] = False  #列打钩

        # Step 6
        # Add the value found in Step 4 to every element of
        # each covered row, and subtract it from every element
        # of each uncovered column.  Return to Step 4 without
        # altering any stars, primes, or covered lines.

        m = min(_uncovered_values(matrix, row_cover, col_cover))
        for r, row in enumerate(matrix):
            for c, __ in enumerate(row):
                if row_cover[r]:  #划线的行
                    matrix[r][c] += m
                if not col_cover[c]:  #未划线的列
                    matrix[r][c] -= m
    # end of _cover_zeroes


def _find_noncovered_zero(matrix, row_cover, col_cover):
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            if value == 0 and not row_cover[r] and not col_cover[c]:
                return (r, c)
    else:
        return None


def _uncovered_values(matrix, row_cover, col_cover):
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            if not row_cover[r] and not col_cover[c]:
                yield value


def _find_star_in_col(mask_matrix, c):
    for r, row in enumerate(mask_matrix):
        if row[c] == STARRED:
            return (r, c)
    else:
        return None


def _find_prime_in_row(mask_matrix, r):
    for c, val in enumerate(mask_matrix[r]):
        if val == PRIMED:
            return (r, c)
    else:
        return None


# Functions for data generation, Gaussian IC case
def generate_data(N, num_H, seed=2017):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    X=np.zeros((num_H,N**2,))
    Y=np.zeros((num_H,N))
    for loop in range(num_H):
        CH = np.random.randint(1,100, size = N**2)
        X[loop,:] = np.reshape(CH, (1,N**2), order="F")
        H=np.reshape(X[loop,:], (N,N), order="F")
        h = H.tolist()
        Y[loop,:] = minimize(h, deepcopy=True)
    return X, Y








