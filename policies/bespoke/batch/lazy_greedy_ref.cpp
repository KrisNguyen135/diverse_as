// C++ implementation of the
// priority queue in which elements
// are sorted by the second element

#include <iostream>
#include <queue>
#include <vector>
#include <math.h>
#include "mex.h"

using namespace std;

typedef pair<int, float> pd;

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

// Function to show the elements
// of the priority queue
void showpq(
	priority_queue<pd,
				vector<pd>, myComp>
		g)
{
	// Loop to print the elements
	// until the priority queue
	// is not empty
	while (!g.empty()) {
		cout << g.top().first
			<< " " << g.top().second
			<< endl;
		g.pop();
	}
	cout << endl;
}

float construct_greedy_batch(float **probs, float loggend[], float jensen_utility, int batch_size, float init_score[], int n, int num_pos_classes)
{
    // create and occupy the priority queue
    priority_queue<pd, vector<pd>, myComp> pq;
    for (int i = 0; i < n; i++)
    {
        pq.push(make_pair(i, init_score[i]));
    }

    // build the batch
    cout << batch_size << endl;
    int batch_ind[batch_size];
    pd next_pair;
    float next_marginal_utility;

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
        jensen_utility = 0;
        for (int j = 0; j < num_pos_classes; j++)
        {
            loggend[j] = loggend[j] + probs[next_pair.first][j];
            jensen_utility = jensen_utility + log(loggend[j]);
        }
    }

    return jensen_utility;
}

// Driver Code
int main()
{
	float probs[4][2] = {
	    {0.4, 0.5},
	    {0.6, 0.2},
	    {0.1, 0.8}
	};
	float loggend[2] = {4.2, 4.7};
	float jensen_utility = log(4.2) + log(4.7);
	int batch_size = 2;
	float init_score[3] = {0.21356, 0.20171, 0.22581};
	int n = 3;
	int num_pos_classes = 2;

	// float jensen_utility = construct_greedy_batch(probs, loggend, jensen_utility, batch_size, init_score, n, num_pos_classes);

	// create and occupy the priority queue
    priority_queue<pd, vector<pd>, myComp> pq;
    for (int i = 0; i < n; i++)
    {
        pq.push(make_pair(i, init_score[i]));
    }

    // build the batch
    int batch_ind[batch_size];
    pd next_pair;
    float next_marginal_utility;

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
        cout << next_pair.first << " " << next_marginal_utility << endl;
        jensen_utility = 0;
        for (int j = 0; j < num_pos_classes; j++)
        {
            loggend[j] = loggend[j] + probs[next_pair.first][j];
            jensen_utility = jensen_utility + log(loggend[j]);
        }
    }

	cout << jensen_utility<< endl;

	return 0;
}
