// C++ implementation of the
// priority queue in which elements
// are sorted by the second element

#include <iostream>
#include <queue>
#include <vector>
#include <math.h>
#include "mex.h"

using namespace std;

typedef pair<int, double> pd;

#define PROBS_ARG           prhs[0]
#define LOGGEND_ARG         prhs[1]
#define JENSEN_UTILITY_ARG  prhs[2]
#define BATCH_SIZE_ARG      prhs[3]
#define INIT_SCORE_ARG      prhs[4]
#define N_ARG               prhs[5]
#define NUM_POS_CLASSES_ARG prhs[6]

#define UTILITY_ARG         plhs[0]
#define ALL_ARG 				    plhs[1]

// Structure of the condition
// for sorting the pair by its
// second elements
struct myComp {
	constexpr bool operator()(
		pd const& a,
		pd const& b)
		const noexcept
	{
		return a.second < b.second;
	}
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
		double **probs;
		double *loggend, jensen_utility, *init_score;
		int batch_size, n, num_pos_classes;

		// PROBS_ARG = mxCreateDoubleMatrix (mxGetM (PROBS_ARG),
    //                               		mxGetN (PROBS_ARG),
    //                               		mxREAL);
		//
		// probs = (double **)mxGetData(PROBS_ARG);

		probs = (double **)mxCalloc(mxGetN(PROBS_ARG), sizeof(double *));

		for (int x = 0; x < mxGetN(PROBS_ARG); x++)
		{
				probs[x] = (double *) mxCalloc(mxGetM(PROBS_ARG), sizeof(double));
		}

		for (int col = 0; col < mxGetN(PROBS_ARG); col++)
		{
				for (int row = 0; row < mxGetM(PROBS_ARG); row++)
				{
						probs[col][row] = mxGetPr(PROBS_ARG)[row + col * mxGetM(PROBS_ARG)];
				}
		}

		loggend = mxGetPr(LOGGEND_ARG);
		jensen_utility = mxGetScalar(JENSEN_UTILITY_ARG);
		batch_size = (int)mxGetScalar(BATCH_SIZE_ARG);
		init_score = mxGetPr(INIT_SCORE_ARG);
		n = (int)mxGetScalar(N_ARG);
		num_pos_classes = (int)mxGetScalar(NUM_POS_CLASSES_ARG);

		// mexPrintf("probs: %.1f %.1f\n", probs[0][0], probs[0][1]);
		// mexPrintf("probs: %.1f %.1f\n", probs[1][0], probs[1][1]);
		// mexPrintf("probs: %.1f %.1f\n", probs[2][0], probs[2][1]);
		//
		// mexPrintf("loggend: %.1f %.1f\n", loggend[0], loggend[1]);
		// mexPrintf("jensen utility: %.4f\n", jensen_utility);
		// mexPrintf("batch size: %d\n", batch_size);
		// mexPrintf("initial scores: %.4f %.4f %.4f\n", init_score[0], init_score[1], init_score[2]);
		// mexPrintf("n: %d\n", n);
		// mexPrintf("number of positive classes: %d\n", num_pos_classes);

		// create and occupy the priority queue
    priority_queue<pd, vector<pd>, myComp> pq;
    for (int i = 0; i < n; i++)
    {
        pq.push(make_pair(i, init_score[i]));
    }

    // build the batch
    int batch_ind[batch_size];
    pd next_pair;
    double next_marginal_utility;

    for (int i = 0; i < batch_size; i++)
    {
        while (true)
        {
            // compute the score of the next point in the queue
            next_pair = pq.top();
            pq.pop();
            next_marginal_utility = 0;
            for (int j = 0; j < num_pos_classes; j++)
            {
                next_marginal_utility = next_marginal_utility + log(loggend[j] + probs[next_pair.first][j]);
            }
            next_marginal_utility = next_marginal_utility - jensen_utility;  // marginal utility

            if (next_marginal_utility >= pq.top().second)  // short-circuit the search
            {
                break;
            }
            else
            {
                pq.push(make_pair(next_pair.first, next_marginal_utility));
            }
        }

        batch_ind[i] = next_pair.first;
        // cout << next_pair.first << " " << next_marginal_utility << endl;
        jensen_utility = 0;
        for (int j = 0; j < num_pos_classes; j++)
        {
            loggend[j] = loggend[j] + probs[next_pair.first][j];
            jensen_utility = jensen_utility + log(loggend[j]);
        }
    }

		UTILITY_ARG = mxCreateDoubleScalar(jensen_utility);
}
