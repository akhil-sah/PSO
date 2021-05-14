#ifndef PSOalgo_H_
#define PSOalgo_H_

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <time.h>
#include<float.h>
#include<math.h>

// Constants
#define PSO_MAX_SWARM_SIZE 100
#define PSO_INERTIA 0.7298

// --- Neighbourhood schemes ---

// gobal best topology
#define PSO_NBD_GLOBAL 0

//ring topology
#define PSO_NBD_RING 1

//random nbd topology
#define PSO_NBD_RANDOM 2

// --- Inertia Weight update strategy ---
#define PSO_W_CONST 0       //constant weight
#define PSO_W_LIN_DEC 1     //linearly decreasing weight

//PSO solution
typedef struct{
	double error;
	double *gbest;
} pso_result;

// Objective function type
typedef double (*pso_obj_fun)(double *, int, void *);

typedef struct{
    int dim;            //problem dimensionality
    double *range_lo;   //lower range limit (array of length DIM)
    double *range_hi;   //higher range limit (array of length DIM)
    double goal;        //optimization goal (error threshold)

    int size;           //swarm size (number of particles)
    int print_every;    //... N steps (set to 0 for no output)
    int steps;          //maximum number of iterations
    int step;           //current PSO step
    double c1;          //cognitive coefficient
    double c2;          //social coefficient
    double w_max;       //max inertia weight value
    double w_min;       //min inertia weight value

    int clamp_pos;     //whether to keep particle position within defined bounds (TRUE)
    					//or apply periodic boundary conditions (FALSE)
    int nhood_strategy; //neighborhood strategy
    int nhood_size;     //neighborhood size
    int w_strategy;     //inertia weight strategy
} pso_settings;

//function to dynamically allocate space for swarm
pso_settings *pso_settings_new(int dim, double range_lo, double range_hi);

//function to free the space alloted
void pso_settings_free(pso_settings *settings);

// return the swarm size based on dimensionality
int pso_swarm_size(int dim);

// minimize the provided obj_fun using PSO with the specified settings
// and store the result in *solution
void pso_solve(pso_obj_fun obj_fun, void *obj_fun_params,
		       pso_result *solution, pso_settings *settings);


//generates a double between (0, 1)
#define RNG_UNIFORM() (rand()/(double)RAND_MAX)

//generates an int between 0 and s (exclusive)
#define RNG_UNIFORM_INT(s) (rand()%s)

//function type for the different inform functions
typedef void (*inform_fun_type)(int *comm, double **pos_nbd, double **pos_b, double *fit_b, double *gbest,
                                int improved, pso_settings *settings);

//function type for the different inertia calculation functions
typedef double (*inertia_fun_type)(int step, pso_settings *settings);

//calulate swarm size based on dimensionality
int pso_swarm_size(int dim)
{
    int size = 10. + 2. * sqrt(dim);
    return (size > PSO_MAX_SWARM_SIZE ? PSO_MAX_SWARM_SIZE : size);
}


//linearly decreasing inertia weight
double inertia_lin_dec(int step, pso_settings *settings)
{
    int dec_stage = 3 * settings->steps / 4;
    if (step <= dec_stage)
        return settings->w_min + (settings->w_max - settings->w_min) * (dec_stage - step) / dec_stage;
    else
        return settings->w_min;
}


// global neighborhood
void inform_global(int *comm, double **pos_nbd, double **pos_b, double *fit_b, double *gbest, int improved,
                   pso_settings *settings)
{
    int i;
    // all particles have the same attractor (gbest)
    // copy the contents of gbest to pos_nbd
    for (i = 0; i < settings->size; i++)
        memmove((void *)pos_nbd[i], (void *)gbest, sizeof(double) * settings->dim);
}

//general inform function :: according to the connectivity
//matrix COMM, it copies the best position (from pos_b) of the
//informers of each particle to the pos_nbd matrix
void inform(int *comm, double **pos_nbd, double **pos_b, double *fit_b, int improved, pso_settings * settings)
{
    int i, j;
    int b_n; // best neighbor in terms of fitness
    // for each particle
    for (j = 0; j < settings->size; j++)
    {
        b_n = j; // self is best
        // to find best informer
        for (i = 0; i < settings->size; i++)
            // the i^th particle informs the j^th particle
            if (comm[i*settings->size + j] && fit_b[i] < fit_b[b_n])
                // found a better informer for j^th particle
                b_n = i;
        // copy pos_b of b_n^th particle to pos_nbd[j]
        memmove((void *)pos_nbd[j], (void *)pos_b[b_n], sizeof(double) * settings->dim);
    }
}

// ring topology
// topology initialization
void init_comm_ring(int *comm, pso_settings * settings)
{
    int i;
    // reset array (initialize with zero)
    memset((void *)comm, 0, sizeof(int) * settings->size * settings->size);

    // choose informers
    for (i = 0; i < settings->size; i++)
    {
        // set diagonal to 1
        comm[i*settings->size + i] = 1;
        if (i==0)
        {
            // look right
            comm[i*settings->size + (i+1)] = 1;    //for making ring [0][1]
            // look left
            comm[(i+1)*settings->size - 1] = 1;    //for making ring  [0][size - 1]
        }
        else if (i == settings->size - 1)     // for last row
        {
            // look right
            comm[i*settings->size] = 1;
            // look left
            comm[i*settings->size + (i-1)] = 1;
        }
        else    // for middle rows
        {
            // look right
            comm[i*settings->size + (i+1)] = 1;
            // look left
            comm[i*settings->size + (i-1)] = 1;
        }
    }
}


void inform_ring(int *comm, double **pos_nbd, double **pos_b, double *fit_b, double *gbest, int improved,
                 pso_settings * settings)
{
    // update pos_nbd matrix
    inform(comm, pos_nbd, pos_b, fit_b, improved, settings);
}

// random neighborhood topology
// topology initialization
void init_comm_random(int *comm, pso_settings * settings)
{

    int i, j, k;
    // reset array
    memset((void *)comm, 0, sizeof(int) * settings->size * settings->size);

    // choose informers
    for (i = 0; i < settings->size; i++)
    {
        // each particle informs itself
        comm[i*settings->size + i] = 1;
        // choose kappa (on average) informers for each particle
        for (k = 0; k < settings->nhood_size; k++)
        {
            // generate a random index
            j = RNG_UNIFORM_INT(settings->size);
            // particle i informs particle j
            comm[i*settings->size + j] = 1;
        }
    }
}

void inform_random(int *comm, double **pos_nbd, double **pos_b, double *fit_b, double *gbest, int improved,
                   pso_settings * settings)
{
    if (!improved)
        init_comm_random(comm, settings);
    inform(comm, pos_nbd, pos_b, fit_b, improved, settings);
}


//create pso settings
pso_settings *pso_settings_new(int dim, double range_lo, double range_hi)
{
    pso_settings *settings = (pso_settings *)malloc(sizeof(pso_settings));
    if (settings == NULL)
        return NULL;

    // set some default values
    settings->dim = dim;
    settings->goal = 1e-5;

    // set up the range arrays
    settings->range_lo = (double *)malloc(settings->dim * sizeof(double));
    if (settings->range_lo == NULL)
    {
        free(settings);
        return NULL;
    }

    settings->range_hi = (double *)malloc(settings->dim * sizeof(double));
    if (settings->range_hi == NULL)
    {
        free(settings);
        free(settings->range_lo);
        return NULL;
    }

    for (int i = 0; i < settings->dim; i++)
    {
        settings->range_lo[i] = range_lo;
        settings->range_hi[i] = range_hi;
    }

    settings->size = pso_swarm_size(settings->dim);
    settings->print_every = 1000;
    settings->steps = 100000;
    settings->c1 = 1.496;
    settings->c2 = 1.496;
    settings->w_max = PSO_INERTIA;
    settings->w_min = 0.3;

    settings->clamp_pos = 1;
    settings->nhood_strategy = PSO_NBD_RING;
    settings->nhood_size = 5;
    settings->w_strategy = PSO_W_LIN_DEC;

    return settings;
}

// destroy PSO settings
void pso_settings_free(pso_settings *settings)
{
    free(settings->range_lo);
    free(settings->range_hi);
    free(settings);
}


double **pso_matrix_new(int size, int dim)
{
    double **m = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
        m[i] = (double *)malloc(dim * sizeof(double));
    return m;
}

void pso_matrix_free(double **m, int size)
{
    for (int i=0; i<size; i++)
        free(m[i]);
    free(m);
}


//==============================================================
//                     PSO ALGORITHM
//==============================================================
void pso_solve(pso_obj_fun obj_fun, void *obj_fun_params, pso_result *solution, pso_settings *settings)
{
    // Particles
    double **pos = pso_matrix_new(settings->size, settings->dim);       // position matrix
    double **vel = pso_matrix_new(settings->size, settings->dim);       // velocity matrix
    double **pos_b = pso_matrix_new(settings->size, settings->dim);     // best position matrix
    double *fit = (double *)malloc(settings->size * sizeof(double));
    double *fit_b = (double *)malloc(settings->size * sizeof(double));
    // Swarm
    double **pos_nbd = pso_matrix_new(settings->size, settings->dim);    // what is best informed
    // position for each particle
    int *comm = (int *)malloc(settings->size * settings->size * sizeof(int));
    // rows : those who inform
    // cols : those who are informed
    int improved = 0; // to keep record whether error was improved in the last iteration

    int i, d, step;
    double a, b; // for matrix initialization
    double rho1, rho2; // random numbers (coefficients)
    // initialize omega using standard value
    double w = PSO_INERTIA;
    inform_fun_type inform_fun = NULL;         // neighborhood update function
    inertia_fun_type calc_inertia_fun = NULL;  // inertia weight update function

    // initialize random seed
    srand(time(NULL));

    // Select appropriate neighbourhood update function
    switch (settings->nhood_strategy)
    {
        case PSO_NBD_GLOBAL:
            // comm matrix not used
            inform_fun = inform_global;
            break;
        case PSO_NBD_RING:
            init_comm_ring(comm, settings);
            inform_fun = inform_ring;
            break;
        case PSO_NBD_RANDOM:
            init_comm_random(comm, settings);
            inform_fun = inform_random;
            break;
        default:
            // use global as the default
            inform_fun = inform_global;
            break;
    }

    // Select appropriate inertia weight update function
    switch (settings->w_strategy)
    {
            /* case PSO_W_CONST : */
            /*     calc_inertia_fun = calc_inertia_const; */
            /*     break; */
        case PSO_W_LIN_DEC :
            calc_inertia_fun = inertia_lin_dec;
            break;
    }

    //Initialise error threshold
    solution->error = DBL_MAX;

    // Swarm initialization for each particle
    for (i = 0; i < settings->size; i++)
    {
        //for each dimension
        for (d = 0; d < settings->dim; d++)
        {
            //generate two numbers within the specified range
            a = settings->range_lo[d] + (settings->range_hi[d] - settings->range_lo[d]) *  RNG_UNIFORM();
            b = settings->range_lo[d] + (settings->range_hi[d] - settings->range_lo[d]) *  RNG_UNIFORM();
            //initialize position
            pos[i][d] = a;
            //best position is the same
            pos_b[i][d] = a;
            //initialize velocity
            vel[i][d] = (a-b) / 2.;
        }
        //update particle fitness
        fit[i] = obj_fun(pos[i], settings->dim, obj_fun_params);
        fit_b[i] = fit[i]; //this is also the personal best
        //update gbest if better solution obtained
        if (fit[i] < solution->error)
        {
            // update best fitness
            solution->error = fit[i];
            // copy particle pos to gbest vector
            memmove((void *)solution->gbest, (void *)pos[i], sizeof(double) * settings->dim);
        }
    }

    //Run algorithm
    for (step = 0; step < settings->steps; step++)
    {
        //update current step
        settings->step = step;
        //update inertia weight
        if (calc_inertia_fun != NULL)
            w = calc_inertia_fun(step, settings);
        //check optimization goal
        if (solution->error <= settings->goal)
        {
            // SOLVED!!
            if (settings->print_every)
                printf("Goal achieved @ step %d (error=%.3e) :-)\n", step, solution->error);
            break;
        }

        //update pos_nbd matrix (find best of neighborhood for all particles)
        inform_fun(comm, (double **)pos_nbd, (double **)pos_b, fit_b, solution->gbest, improved, settings);
        //the value of improved was just used; reset it
        improved = 0;

        //update all particles
        for (i = 0; i < settings->size; i++)
        {
            //for each dimension
            for (d = 0; d < settings->dim; d++)
            {
                //stochastic coefficients
                rho1 = settings->c1 * RNG_UNIFORM();
                rho2 = settings->c2 * RNG_UNIFORM();
                //update velocity
                vel[i][d] = w * vel[i][d] + rho1 * (pos_b[i][d] - pos[i][d]) + rho2 * (pos_nbd[i][d] - pos[i][d]);
                //update position
                pos[i][d] += vel[i][d];
                //clamp position within bounds
                if (settings->clamp_pos)
                {
                    if (pos[i][d] < settings->range_lo[d])
                    {
                        pos[i][d] = settings->range_lo[d];
                        vel[i][d] = 0;
                    }
                    else if (pos[i][d] > settings->range_hi[d])
                    {
                        pos[i][d] = settings->range_hi[d];
                        vel[i][d] = 0;
                    }
                }
                else
                {
                    // enforce periodic boundary conditions
                    if (pos[i][d] < settings->range_lo[d])
                    {
                        pos[i][d] = settings->range_hi[d] - fmod(settings->range_lo[d] - pos[i][d],
                                                                 settings->range_hi[d] - settings->range_lo[d]);
                        vel[i][d] = 0;
                    }
                    else if (pos[i][d] > settings->range_hi[d])
                    {
                        pos[i][d] = settings->range_lo[d] + fmod(pos[i][d] - settings->range_hi[d],
                                                                 settings->range_hi[d] - settings->range_lo[d]);
                        vel[i][d] = 0;
                    }
                }

            }

            //update particle fitness
            fit[i] = obj_fun(pos[i], settings->dim, obj_fun_params);
            //update personal best position
            if (fit[i] < fit_b[i])
            {
                fit_b[i] = fit[i];
                //copy contents of pos[i] to pos_b[i]
                memmove((void *)pos_b[i], (void *)pos[i], sizeof(double) * settings->dim);
            }
            //update gbest
            if (fit[i] < solution->error)
            {
                improved = 1;
                //update best fitness
                solution->error = fit[i];
                //copy particle pos to gbest vector
                memmove((void *)solution->gbest, (void *)pos[i], sizeof(double) * settings->dim);
            }
        }

        if (settings->print_every && (step % settings->print_every == 0))
            printf("Step %d (w=%.2f) :: min err=%.5e\n", step, w, solution->error);
    }

    // free resources
    pso_matrix_free(pos, settings->size);
    pso_matrix_free(vel, settings->size);
    pso_matrix_free(pos_b, settings->size);
    pso_matrix_free(pos_nbd, settings->size);
    free(comm);
    free(fit);
    free(fit_b);
}


#endif // PSO_H_

