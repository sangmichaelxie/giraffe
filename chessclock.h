/*
	Copyright (C) 2015 Matthew Lai

	Giraffe is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	Giraffe is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CHESSCLOCK_H
#define CHESSCLOCK_H

#include <cstdint>

class ChessClock
{
public:
	enum Mode
	{
		CONVENTIONAL_INCREMENTAL_MODE,
		EXACT_MODE // exact time per move
	};

	ChessClock(Mode mode, int numMoves, double baseTime, double inc);

	void Reset(); //default stopped
	double GetReading() const ; //time left in seconds
	Mode GetMode() const { return m_mode; }
	double GetInc() const { return m_inc; }
	void Start();
	void Stop(); //pause
	void MoveMade(); //notify the clock that a move by the associated side is made (increases timer accordingly)
	void AdjustTime(double time);

	int32_t GetMovesUntilNextPeriod() const { return (m_numMoves == 0) ? 0 : (m_numMoves - m_numMovesMade); }

private:
	//initial parameters (used by reset)
	Mode m_mode;
	int m_numMoves;
	double m_baseTime; // time per period
	double m_inc;

	int m_numMovesMade;
	double m_endTime;
	double m_timeLeftWhenStopped;
	bool m_running;
};

#endif // CHESSCLOCK_H
