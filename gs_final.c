/* CSCI-GA 3033-016 Multicore Processors
 * Lab 2 - Solving Linear Equations using MPI
 * Submitted by: Xialiang Liu (N11861210)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>

/*** Rewritten from Skeleton for Lab 2 ***/

/* =================
 * Global variables
 * ================= 
 */
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */
// new added:
bool isAllDone = false; /* a flag indicating the solution is final */
bool isUsingMPI = true; /* whether we use MPI/run in parallel */

/* ======================
 * Function declarations 
 * ======================
 */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */
float getError(float new_x, float old_x); /* Compute absolute relative error */
void updateNewSetXs(); /* Generate a new set of Xs */
int runSequential(); /* Sequential execution driver */
int runParallel(int comm_size, int my_rank); /* Parallel execution driver */
void freeMemAlloc(); /* Release resource before finish */
void writeSolution(); /* Write solution to text file */

/* ======================
 * Functions
 * ======================
 */
float getError(float new_x, float old_x)
{
    /* A small function that compute the absolute relative error */
    return fabs((new_x - old_x) / new_x);
}

void updateNewSetXs()
{
    /* This is the main function that execute the iteration 
     * of computation for a new set of Xs.
     */
    float* newX_i = (float *) malloc(num * sizeof(float));
    float* newErr = (float *) malloc(num * sizeof(float));

    // Compute the values for x_i in this iteration
    int i, j;
    for (i = 0; i < num; ++i)
    {
        float numerator = b[i];
        for (j = 0; j < num; ++j)
        {
            if (j != i){
                numerator -= (a[i][j] * x[j]);
            }
        }
        float x_i = numerator / a[i][i];

        newX_i[i] = x_i;
    }

    // Evalidate the error margin for each x_i
    bool isDone = true;
    for (i = 0; i < num; ++i)
    {
        newErr[i] = fabs((newX_i[i] - x[i]) / (newX_i[i]));
        if (newErr[i] > err){
            isDone = false;
        }
        x[i] = newX_i[i];
    }

    if (isDone){
        isAllDone = true;
    }

    // release resources for temporary variables
    free(newX_i);
    free(newErr);
}

int runSequential()
{
    /* This is the driver function for sequential mode */
    int num_iter = 0;
    for (; !isAllDone; ++num_iter)
    {
        updateNewSetXs();
        if (isAllDone) break;
    }
    return num_iter;
}

int runParallel(int comm_size, int my_rank)
{
    /* This is the driver function of running in parallel mode.
     * It assigns jobs for each process.
     */
    int num_iter = 0; // counter for the number of iterations
    int local_size;      // size of local work for each process

    int upper_index, lower_index;
    float * local_new;
    int recv_counts[comm_size];  // integer array of receive counts from each processes
    int recv_offsets[comm_size]; // integer array of the beginning index for  incoming data from each processes

    int step = num / comm_size; // use step to keep the value of floor(num / comm_size)

    int i;
    
    // WE ONLY HANDLE THE CASE THAT #PROCESS <= #UNKNOWNS
    if (num % comm_size){
        // if #unknown is not a multiple of #process
        int offset = (num % comm_size) * (step + 1);
        
        for (i = 0; i < comm_size; i++){
            if (i < (num % comm_size)){
                recv_counts[i] = step + 1;
                recv_offsets[i] = (step + 1) * i;
            }
            else {
                recv_counts[i] = step;
                recv_offsets[i] = offset + (i - (num % comm_size)) * step;
            }
            if (my_rank == i){
                local_size = recv_counts[i];
                lower_index = recv_offsets[i];
                upper_index = lower_index + local_size;
            }
        }
    }
    else {
        // When #unknown is divisible by #processes, life is easy
        local_size = num / comm_size;
        lower_index = local_size * my_rank;
        upper_index = local_size * (my_rank + 1);
        
        for (i = 0; i < comm_size; i++){
            recv_counts[i] = local_size;
            recv_offsets[i] = local_size * i;
        }
    }
    // For DEBUG USE only:
    //printf("local size, lower, upper: %d, %d, %d\n", local_size, lower_index, upper_index);

    // Initialize the buffer containing all Xs which 
    // will be updated and received by all processes
    float * all_new = (float *) malloc(num * sizeof(float));
    float * new_errors = (float *) malloc(num * sizeof(float));
    memset(new_errors, 0, sizeof(float) * num);
    memset(all_new, 0, sizeof(float) * num);

    // Compute the assigned work
    local_new = (float *) malloc(local_size * sizeof(float));

    for (; !isAllDone; ++num_iter)
    {
        if (my_rank < num)
        {
            int local_index = 0;
            for (i = lower_index; i < upper_index; ++i)
            {
                // Solve x_i
                local_new[local_index] = b[i];

                int j;
                for (j = 0; j < num; ++j)
                {
                    if (j != i){
                        local_new[local_index] -= (a[i][j] * x[j]);
                    }
                }
                local_new[local_index] /= a[i][i];
                ++local_index;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgatherv(
                local_new, //send_buffer
                local_size, // send_count
                MPI_FLOAT, // send_datatype
                all_new,  // recv_buffer
                recv_counts, // array of recv_counts for all processes
                recv_offsets, // array of recv_offsets
                MPI_FLOAT,  // recv_datatype
                MPI_COMM_WORLD); // communicator
        MPI_Barrier(MPI_COMM_WORLD);

        // Each process checks whether each x_i meets the error margin condition
        bool isDone = true;
        for (i = 0; i < num; ++i)
        {
            new_errors[i] = fabs((all_new[i] - x[i]) / (all_new[i]));
            if (new_errors[i] > err){
                isDone = false;
            }
            x[i] = all_new[i];
        }

        if (isDone){
            isAllDone = true;
        }
    }

    free(new_errors);
    free(all_new);
    free(local_new);

    return num_iter;
}

void check_matrix()
{
    /* Check whether the coefficient matrix is convergent. 
     * Conditions for convergence (diagonal dominance):
     * 1. diagonal element >= sum of all other elements of the row
     * 2. At least one diagonal element > sum of all other elements of the row
     */
    int bigger = 0; /* Set to 1 if at least one diag element > sum  */
    int i, j;
    float sum = 0;
    float aii = 0;
    
    for(i = 0; i < num; i++)
    {
        sum = 0;
        aii = fabs(a[i][i]);
        
        for(j = 0; j < num; j++)
            if( j != i)
                sum += fabs(a[i][j]);
        
        if( aii < sum)
        {
            printf("The matrix will not converge.\n");
            exit(1);
        }
        
        if(aii > sum)
            bigger++;
    }
    
    if( !bigger )
    {
        printf("The matrix will not converge\n");
        exit(1);
    }
}



void get_input(char filename[])
{
    /* Read input from file provided in the skeleton. */
    /* After this function returns:
     * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
     * x[] will contain the initial values of x
     * b[] will contain the constants (i.e. the right-hand-side of the equations
     * num will have number of variables
     * err will have the absolute error that you need to reach
     */
    FILE * fp;
    int i, j;
    
    fp = fopen(filename, "r");
    if(!fp)
    {
        printf("Cannot open file %s\n", filename);
        exit(1);
    }
    
    fscanf(fp,"%d ",&num);
    fscanf(fp,"%f ",&err);
    
    /* Now, time to allocate the matrices and vectors */
    a = (float**)malloc(num * sizeof(float*));
    if( !a)
    {
        printf("Cannot allocate a!\n");
        exit(1);
    }
    
    for(i = 0; i < num; i++) 
    {
        a[i] = (float *)malloc(num * sizeof(float)); 
        if( !a[i])
        {
            printf("Cannot allocate a[%d]!\n",i);
            exit(1);
        }
    }
    
    x = (float *) malloc(num * sizeof(float));
    if( !x)
    {
        printf("Cannot allocate x!\n");
        exit(1);
    }

    b = (float *) malloc(num * sizeof(float));
    if(!b)
    {
        printf("Cannot allocate b!\n");
        exit(1);
    }
    
    /* Now .. Filling the blanks */
    
    /* The initial values of Xs */
    for(i = 0; i < num; i++)
        fscanf(fp,"%f ", &x[i]);
    
    for(i = 0; i < num; i++)
    {
        for(j = 0; j < num; j++)
            fscanf(fp,"%f ",&a[i][j]);
        
        /* reading the b element */
        fscanf(fp,"%f ",&b[i]);
    }
    fclose(fp);
}

void writeSolution()
{
    /* Generate the x.sol file where x is the number of unknowns. */
    FILE * fp;
    char output[100] = "";

    /* Write results to file */
    sprintf(output,"%d.sol", num);
    fp = fopen(output,"w");
    if (!fp)
    {
        printf("Sorry! Cannot create the file %s\n", output);
        exit(1);
    }

    int i;
    for (i = 0; i < num; i++)
    {
        fprintf(fp, "%f\n", x[i]);
    }
    fclose(fp);
}

void freeMemAlloc()
{
    /* A clean-up step before finish the program. */
    free(x);
    free(b);
    int i;
    for (i = 0; i < num; ++i){
        free(a[i]);
    }
    free(a);
}


int main(int argc, char *argv[])
{
    
    int nIteration = 0; /* total number of iterations */
    //FILE * fp;
    //char output[100] ="";
    
    if( argc != 2)
    {
        printf("Usage: ./<output executable file> <input file>\n");
        exit(1);
    }
    
    /* Read the input file and fill the global data structure above */ 
    get_input(argv[1]);
    
    /* Check for convergence condition */
    /* This function will exit the program if the coffeicient will never 
     * converge to the needed absolute error. 
     * This is not expected to happen for this programming assignment.
     */
    check_matrix();

    int comm_size;
    int my_rank;

    if (isUsingMPI)
    {
        // Initialize MPI 
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        // In this assignment, we only deal with the cases #process <= #unknown
        if (comm_size > num){
            if (my_rank == 0)
                printf("Sorry! Please allow more unknowns than processes!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // get the number of iterations to solve problem
        nIteration = runParallel(comm_size, my_rank);
        if (my_rank == 0)
        { 
            // only write output and print number of iterations in one process
            writeSolution();
            printf("total number of iterations: %d\n", nIteration);
        }
        MPI_Finalize();
    }
    else 
    {
        nIteration = runSequential();
        writeSolution();
        printf("total number of iterations: %d\n", nIteration);
    }

    freeMemAlloc();
    exit(0);
}
