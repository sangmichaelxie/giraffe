#include "history.h"

#include <cmath>

History::History()
{
	for (int32_t c = 0; c < 2; ++c)
	{
		for (int32_t from = 0; from < 64; ++from)
		{
			for (int32_t to = 0; to < 64; ++to)
			{
				m_cutoffCounts[c][from][to] = 0;
				m_nonCutoffCounts[c][from][to] = 0;
			}
		}
	}
}

void History::NotifyCutoff(Move move, NodeBudget nodeBudget)
{
	Square from = GetFromSquare(move);
	Square to = GetToSquare(move);
	Color c = GetColor(GetPieceType(move));

	m_cutoffCounts[c == WHITE ? 0 : 1][from][to] += std::pow(std::log(static_cast<float>(nodeBudget)), 2);
}

void History::NotifyNoCutoff(Move move, NodeBudget nodeBudget)
{
	Square from = GetFromSquare(move);
	Square to = GetToSquare(move);
	Color c = GetColor(GetPieceType(move));

	m_nonCutoffCounts[c == WHITE ? 0 : 1][from][to] += std::pow(std::log(static_cast<float>(nodeBudget)), 2);
}

float History::GetHistoryScore(Move move)
{
	Square from = GetFromSquare(move);
	Square to = GetToSquare(move);
	Color c = GetColor(GetPieceType(move));

	uint64_t posScore = m_cutoffCounts[c == WHITE ? 0 : 1][from][to];
	uint64_t negScore = m_nonCutoffCounts[c == WHITE ? 0 : 1][from][to];

	if ((posScore + negScore) == 0)
	{
		return 0.5f;
	}
	else
	{
		return static_cast<float>(posScore) / (posScore + negScore);
	}
}

void History::NotifyMoveMade()
{
	for (int32_t c = 0; c < 2; ++c)
	{
		for (int32_t from = 0; from < 64; ++from)
		{
			for (int32_t to = 0; to < 64; ++to)
			{
				m_cutoffCounts[c][from][to] /= 2;
				m_nonCutoffCounts[c][from][to] /= 2;
			}
		}
	}
}
