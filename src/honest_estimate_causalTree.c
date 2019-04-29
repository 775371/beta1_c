/* 
 * Do honest causalTree estimation with parameters
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double** matrix(int n, int m) {
	double **X;
	X = malloc(n * sizeof(*X));
	for (int i = 0; i < n; i++)
		X[i] = malloc(m * sizeof(*X[i]));
	return X;
}

void clear(int n, double** X) {
	for (int i = 0; i < n; i++)
		free(X[i]);
	free(X);
}

double** transpose(int n, int m, double** X) {
	double** X_ = matrix(m, n);
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			X_[i][j] = X[j][i];
		}
	}
	return X_;
}

double** product(int n, int m, int p, int q, double** A, double** B) {
	double** C = matrix(n, q);

	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < q; j++)
			C[i][j] = 0;

	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < q; j++) {
			for (size_t k = 0; k < p; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	return C;
}

double** add(int n, int m, int p, int q, double** A, double** B) {
	double** D = matrix(n, q);

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
				D[i][j] = A[i][j] + B[i][j];
		}
	}
	return D;
}


double** sub(int n, int m, int p, int q, double** A, double** B) {
	double** S = matrix(n, m);

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
				S[i][j] = A[i][j] - B[i][j];
		}
	}
	return S;
}



double** identity(int n, int m, double** X) {
	double** I = matrix(n, m);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < m; j++) {
			if (i == j) 
				I[i][j] = 1;
			else I[i][j] = 0;
		}
	}
	return I;
}


double** get_minor(int row, int col, int n, double** M) {
	int k = 0;
	int l = 0;
	int s = n - 1;
	double** m = matrix(s, s);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i != row && j != col) {
				m[k][l] = M[i][j]; l++;
			}
		}
		if (i != row) k++;
		l = 0;
	}
	return m;
}

double determinant(int n, double** M) {
	if (n == 2)
		return (M[0][0] * M[1][1]) - (M[0][1] * M[1][0]);

	double det = 0;
	for (size_t j = 0; j < n; j++) {
		double** m = get_minor(0, j, n, M);
		det += pow((-1), j)*M[0][j] * determinant(n - 1, m);
		clear(n, m);
	}
	return det;
}

double** inverse(int n, double** M) {
	double** C = matrix(n, n);
	double d = determinant(n, M);

	if (n == 2) {
		C[0][0] = M[1][1] / d;
		C[0][1] = (-1)*M[0][1] / d;
		C[1][0] = (-1)*M[1][0] / d;
		C[1][1] = M[0][0] / d;
		return C;
	}

	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			double** m = get_minor(i, j, n, M);
			C[i][j] = (pow((-1), i + j)*determinant(n - 1, m)) / d;
		}
	}

	double** A = transpose(n, n, C);
	clear(n, C);
	return A;
}

void output(int n, int m, double** X, char* T) {
	printf("%s\n---\n", T);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < m; j++) {
			if (j == m - 1) printf("%.2f", X[i][j]);
			else printf("%.2f, ", X[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

/*
* Linear Least Squares Optimization - Normal Equations Approach
*
* @param n: number of input rows
* @param m: number of input columns
* @param X: input matrix
* @param y: output vector
*
* @return: linear weights
*/
double** lstsq(int n, int m, double** X, double** y) {

	// TODO: include bias column

	double** X_ = transpose(n, m, X);
	double** A = product(m, n, n, m, X_, X);

	printf("\n");
	output(n, 1, y, "y");
	output(n, m, X, "X");
	output(m, n, X_, "X^T");
	output(m, m, A, "A=X*X^T");

	// non-square matrices do not have determinant
	double d = determinant(m, A);
	if (d == 0) {
		printf("Matrix non-invertible.\n\n");
		exit(-1);
	}
	else printf("Determinant: %.1f\n\n", d);

	double** A_ = inverse(m, A);
	double** B = product(m, m, m, n, A_, X_);
	double** w = product(m, n, n, 1, B, y);

	output(m, m, A_, "A^T");
	output(m, n, B, "B=A^T*X^T");
	output(m, 1, w, "B*y");

	clear(n, X_);
	clear(n, A);
	clear(n, A_);
	clear(n, B);

	return w;
}


//----------------------end--------------------------------------------------

#include "causalTree.h"
#include "causalTreeproto.h"

    static void
honest_estimate_causalTree0(const int *dimx, int nnode, int nsplit, const int *dimc, 
                            const int *nnum, const int *nodes2, const int *vnum,
                            const double *split2, const int *csplit2, const int *usesur,
                            int *n1, double *wt1, double *dev1, double *yval1, const double *xdata2, 
                            const double *wt2, const double *treatment2, const double *y2,
                            const double *matrix2, //add matrix
                            const int *xmiss2, int *where)
{
    Rprintf("honest_estimate_causalTree.c\n");
    int i, j;
    int n;
    int ncat;
    int node, nspl, var, dir;
    int lcount, rcount;
    int npos;
    double temp;
    const int *nodes[3];
    const double *split[4];
    const int **csplit = NULL, **xmiss;
    const double **xdata;
	    //input 
    const double **matrix;
    const double **tempM;
    const double **tempY;
	    
    double *trs = NULL;
    double *cons = NULL; 
    double *trsums = NULL; 
    double *consums = NULL;
    double *trsqrsums = NULL;
    double *consqrsums = NULL;
    int nnodemax = -1;
    int *invertdx = NULL;
    /*add beta*/
    double  *y_sum= NULL; 
    double  *z_sum =  NULL;
    double *yz_sum = NULL;
    double *yy_sum = NULL;
    double *zz_sum = NULL;
   
    
    trs = (double *) ALLOC(nnode, sizeof(double));
    cons = (double *) ALLOC(nnode, sizeof(double));
    trsums = (double *) ALLOC(nnode, sizeof(double));
    consums = (double *) ALLOC(nnode, sizeof(double));
    trsqrsums = (double *) ALLOC(nnode, sizeof(double));
    consqrsums = (double *) ALLOC(nnode, sizeof(double));
    
    /*add beta*/
    y_sum= (double *) ALLOC(nnode, sizeof(double));
    z_sum = (double *) ALLOC(nnode, sizeof(double));
    yz_sum = (double *) ALLOC(nnode, sizeof(double));
    yy_sum = (double *) ALLOC(nnode, sizeof(double));
    zz_sum = (double *) ALLOC(nnode, sizeof(double));
     
     
     
     
    // add matrix 
     
    // initialize:
    for (i = 0; i < nnode; i++) {
        trs[i] = 0.;
        cons[i] = 0.;
        trsums[i] = 0.;
        consums[i] = 0.;
        trsqrsums[i] = 0.;
        consqrsums[i] = 0.;
     /* add */
     y_sum[i] = 0.;
     z_sum[i] = 0.;
     yz_sum[i] = 0.;
     yy_sum[i] = 0.;
     zz_sum[i] = 0.;
     
     
        n1[i] = 0;
        wt1[i] = 0.;
        if (nnum[i] > nnodemax) {
            nnodemax = nnum[i]; 
        }
    }
    
    invertdx = (int *) ALLOC(nnodemax + 1, sizeof(int));
    // construct an invert index:
    for (i = 0; i <= nnodemax + 1; i++) {
        invertdx[i] = -1;
    }
    for (i = 0; i != nnode; i++) {
        invertdx[nnum[i]] = i;
    }
    
    n = dimx[0]; // n = # of obs
    for (i = 0; i < 3; i++) {
        nodes[i] = &(nodes2[nnode * i]);
    }
    
    for(i = 0; i < 4; i++) {
        split[i] = &(split2[nsplit * i]);
    }

    if (dimc[1] > 0) {
        csplit = (const int **) ALLOC((int) dimc[1], sizeof(int *));
        for (i = 0; i < dimc[1]; i++)
            csplit[i] = &(csplit2[i * dimc[0]]);
    }
    xmiss = (const int **) ALLOC((int) dimx[1], sizeof(int *));
    xdata = (const double **) ALLOC((int) dimx[1], sizeof(double *));
	    // alloacate input matrix
    matrix = (const double **) ALLOC((int) dimx[1], sizeof(double *));
     tempM = (const double **) ALLOC((int) dimx[1], sizeof(double *));
     tempY = (const double **) ALLOC((int) dimx[1], sizeof(double *));
	    
     Rprintf("The dimx  in honest_estimate_causalTree.c is %d\n", dimx);
	    
    for (i = 0; i < dimx[1]; i++) {
        xmiss[i] = &(xmiss2[i * dimx[0]]);
        xdata[i] = &(xdata2[i * dimx[0]]);
	matrix[i] = &(matrix2[i * dimx[0]]);
    
    }
    

    for (i = 0; i < n; i++) {
        node = 1;               /* current node of the tree */
next:
        for (npos = 0; nnum[npos] != node; npos++);  /* position of the node */

        n1[npos]++;
        wt1[npos] += wt2[i];
     
     
        trs[npos] += wt2[i] * treatment2[i];
        cons[npos] += wt2[i] * (1 - treatment2[i]);
        /*trsums[npos] += wt2[i] * treatment2[i] * y2[i];*/
      /* add variable*/
        trsums[npos] += wt2[i]  * y2[i];
       
     
        consums[npos] += wt2[i] * (1 - treatment2[i]) * y2[i];
        /*trsqrsums[npos] +=  wt2[i] * treatment2[i] * y2[i] * y2[i];*/
        trsqrsums[npos] +=  wt2[i]  * y2[i] * y2[i];
     
        consqrsums[npos] += wt2[i] * (1 - treatment2[i]) * y2[i] * y2[i];
        
        y_sum[npos] += wt2[i] * treatment2[i];
        z_sum[npos] += wt2[i] * y2[i];
        yz_sum[npos] += wt2[i] * y2[i] * treatment2[i];
       
        yy_sum[npos] += wt2[i] * treatment2[i] * treatment2[i];
        zz_sum[npos] += wt2[i] * y2[i] * y2[i];
     
        
        /* walk down the tree */
        nspl = nodes[2][npos] - 1;      /* index of primary split */
        if (nspl >= 0) {        /* not a leaf node */
            var = vnum[nspl] - 1;
            if (xmiss[var][i] == 0) {   /* primary var not missing */
                ncat = (int) split[1][nspl];
                temp = split[3][nspl];
                if (ncat >= 2)
                    dir = csplit[(int) xdata[var][i] - 1][(int) temp - 1];
                else if (xdata[var][i] < temp)
                    dir = ncat;
                else
                    dir = -ncat;
                if (dir) {
                    if (dir == -1)
                        node = 2 * node;
                    else
                        node = 2 * node + 1;
                    goto next;
                }
            }
            if (*usesur > 0) {
                for (j = 0; j < nodes[1][npos]; j++) {
                    nspl = nodes[0][npos] + nodes[2][npos] + j;
                    var = vnum[nspl] - 1;
                    if (xmiss[var][i] == 0) {   /* surrogate not missing */
                        ncat = (int) split[1][nspl];
                        temp = split[3][nspl];
                        if (ncat >= 2)
                            dir = csplit[(int)xdata[var][i] - 1][(int)temp - 1];
                        else if (xdata[var][i] < temp)
                            dir = ncat;
                        else
                            dir = -ncat;
                        if (dir) {
                            if (dir == -1)
                                node = 2 * node;
                            else
                                node = 2 * node + 1;
                            goto next;
                        }
                    }
                }
            }
            if (*usesur > 1) {  /* go with the majority */
                for (j = 0; nnum[j] != (2 * node); j++);
                lcount = n1[j];
                for (j = 0; nnum[j] != (1 + 2 * node); j++);
                rcount = n1[j];
                if (lcount != rcount) {
                    if (lcount > rcount)
                        node = 2 * node;
                    else
                        node = 2 * node + 1;
                    goto next;
                }
            }
        }
        where[i] = node;
    }
    
    for (i = 0; i <= nnodemax; i++) {
        if (invertdx[i] == -1)
            continue;
        int origindx = invertdx[i];
     
        //base case
        
          if (trs[origindx] != 0 && cons[origindx] != 0) {
            double tr_mean = trsums[origindx] * 1.0 / wt1[origindx];
            double con_mean = consums[origindx] * 1.0 / cons[origindx];
            double tt_mean =  yy_sum[origindx]* 1.0 / wt1[origindx];
           
            /*yval1[origindx] = tr_mean - con_mean;*/
            /*dev1[origindx] = trsqrsums[origindx] - trs[origindx] * tr_mean * tr_mean 
                + consqrsums[origindx] - cons[origindx] * con_mean * con_mean;*/
	   int m = sizeof(matrix[0]) / sizeof(*matrix[0]);
           //double n = sizeof(matrix) / sizeof(*matrix);
	for ( int j = 0; j < m; j++ ) //row
    {
	     tempM[i][j]= matrix[i][j];
	   
     }       
           tempY[i][0] = tempY[i][0];
          
           double** w = lstsq(n, m, tempM, tempY);
          yval1[origindx] =w[0][0] ;
          dev1[origindx] = trsqrsums[origindx] - trs[origindx] * tr_mean * tr_mean 
                + consqrsums[origindx] - cons[origindx] * con_mean * con_mean;
           
         //  Rprintf("The trsqrsums in  honest.causaltree.c is %d\n", trsqrsums);
          // Rprintf("The consqrsums in  honest.causaltree.c is %d\n", consqrsums);
           
        } else {
            int parentdx = invertdx[i / 2];
            yval1[origindx] = yval1[parentdx];
            dev1[origindx] = yval1[parentdx];
         
        }
    Rprintf("The dev1 in honest.causaltree.c is %d\n", dev1);
    Rprintf("The yval1 in honest.causaltree.c is %d\n", yval1);
    }
    Rprintf("end the honest estimate tree\n");
    
}
   
#include <Rinternals.h>

SEXP
honest_estimate_causalTree(SEXP dimx, SEXP nnode, 
                           SEXP nsplit, SEXP dimc, SEXP nnum, 
                           SEXP nodes2, 
                           SEXP n1, SEXP wt1, SEXP dev1, SEXP yval1, 
                           SEXP vnum, 
                           SEXP split2,
                           SEXP csplit2, SEXP usesur, 
                           SEXP xdata2, SEXP wt2, SEXP treatment2, SEXP y2, SEXP matrix2,
                           SEXP xmiss2)
{
    int n = asInteger(dimx);
    SEXP where = PROTECT(allocVector(INTSXP, n));
    honest_estimate_causalTree0(INTEGER(dimx), asInteger(nnode), asInteger(nsplit),
            INTEGER(dimc), INTEGER(nnum), INTEGER(nodes2),
            INTEGER(vnum), REAL(split2), INTEGER(csplit2),
            INTEGER(usesur), 
            INTEGER(n1), REAL(wt1), REAL(dev1), REAL(yval1), 
            REAL(xdata2), REAL(wt2), REAL(treatment2), REAL(y2), REAL(matrix2),
            INTEGER(xmiss2), INTEGER(where));
    UNPROTECT(1);
    return where;
}
