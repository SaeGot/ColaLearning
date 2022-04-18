#include "ReinforcementLearning.h"


void ReinforcementLearning::DecayEpsilonGreedy(double decay)
{
	epsilonGreeedy *= decay;
}
