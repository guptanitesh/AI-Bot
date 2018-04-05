import copy
import datetime
import time
import random


class Team48:
	def __init__(self):
		self.myMove = False
		self.timeLimit = datetime.timedelta(seconds = 14.5)
		# self.lastCallTime = datetime.timedelta(seconds = 14)
		self.begin = 0
		self.index = 0
		self.marker = 'x'
		self.zobrist = []
		self.heuristicDict = dict()
		self.transpositionTable = {}
		self.moveOrder = {}
		self.INFINITY = 10000000000000
		self.block_won = 100000
		self.one_win = 4000
		self.two_win = 200
		self.three_win = 10
		self.one_win_board = 200000000
		self.two_win_board = 10000000
		self.three_win_board = 400000
		self.allList = tuple([(i, j) for i in xrange(4) for j in xrange(4)])
		self.zobristInitialize()

		# For diamonds
		self.diamond1 = [(1, 0), (0, 1), (2, 1), (1, 2)]
		self.diamond2 = [(1, 1), (0, 2), (2, 2), (1, 3)]
		self.diamond3 = [(2, 0), (1, 1), (3, 1), (2, 2)]
		self.diamond4 = [(2, 1), (1, 2), (3, 2), (2, 3)]


	def allChildNodes(self, formattedBoard, formattedBlockStatus, root):
		moveList = []
		for allowedBlock in self.checkValidBlocks(root, formattedBlockStatus):
			tp = [(i, j) for i in xrange(4) for j in xrange(4) if formattedBoard[allowedBlock[0]][allowedBlock[1]][i][j] == 0]
			moveList += [ (4 * allowedBlock[0] + move[0], 4 * allowedBlock[1] + move[1]) for move in tp]
		return moveList

	def zobristInitialize(self):
		self.zobrist = []
		for i in xrange(16):
			self.zobrist.append([])
			for j in xrange(16):
				self.zobrist[i].append([])
				for k in xrange(17):
					self.zobrist[i][j].append([])
					for l in xrange(17):
						self.zobrist[i][j][k].append([])		
						for m in xrange(2):
							tp = random.randint(0, 2**64)
							self.zobrist[i][j][k][l].append(tp)

	def isTimeLeft(self):
		return datetime.datetime.utcnow() - self.begin > self.timeLimit

	def move(self, currentBoard, oldMove, flag):
		self.marker = flag

		formattedBlockStatus = [[0] * 4 for i in xrange(0, 4)]
		formattedBoard = [[[[0] * 4 for i in xrange(0, 4)] for j in xrange(0, 4)] for k in xrange(0, 4)]

		for r in xrange(4):
			for c in xrange(4):
				# 1->we won, 0->nothing, 3->drawn, 2->other won
				tp = currentBoard.block_status[r][c]
				if tp == 'd':
					formattedBlockStatus[r][c] = 3
				elif tp == '-':
					formattedBlockStatus[r][c] = 0
				elif tp == flag:
					formattedBlockStatus[r][c] = 1
				else:
					formattedBlockStatus[r][c] = 2

		for r in xrange(16):
			for c in xrange(16):
				# 1->us, 0->empty, 2->other
				tp = currentBoard.board_status[r][c]
				e = r/4
				f = c/4
				g = r%4
				h = c%4
				if tp == '-':
					formattedBoard[e][f][g][h] = 0
				else:
					if tp != flag:
						formattedBoard[e][f][g][h] = 2
					else:
						formattedBoard[e][f][g][h] = 1

		
		if oldMove[0] < 0 or oldMove[1] < 0:
			return (11, 11)

		self.moveOrder = {}
		posR = oldMove[0]
		posC = oldMove[1]
		isPlayerBonus = True
		
		if currentBoard.board_status[posR][posC] == flag:
			isPlayerBonus = False

		best_move = self.IDS(formattedBoard, formattedBlockStatus, oldMove, isPlayerBonus)

		return best_move

	def IDS(self, formattedBoard, formattedBlockStatus, root, isPlayerBonus):
		firstGuess = 0

		self.begin = datetime.datetime.utcnow()
		maxDepth = 16 * 16

		for depth in range(1, maxDepth + 1):
			self.transpositionTable = {}
			if self.isTimeLeft():
				break

			firstGuess, move = self.mtdf(formattedBoard, formattedBlockStatus, root, firstGuess, depth, isPlayerBonus)
			# if self.isTimeLeft() or firstGuess == "TIMEOUT":
			if self.isTimeLeft():
				break
			finalMove = move

		# print "defp4 depth: ", depth
		return finalMove

	def mtdf(self, formattedBoard, formattedBlockStatus, root, firstGuess, depth, isPlayerBonus):
		lowerBound, upperBound = -self.INFINITY, self.INFINITY
		
		g = firstGuess

		while lowerBound < upperBound:
			if g == lowerBound:
				beta = g + 1
			else:
				beta = g

			self.myMove, self.index = True, 0
			# if datetime.datetime.utcnow() - self.begin > self.lastCallTime:
			# 	return "TIMEOUT", "TIMEOUT"

			alphaBeta_params = {
				"formattedBoard": formattedBoard,
				"formattedBlockStatus": formattedBlockStatus,
				"root": root,
				"alpha": beta - 1,
				"beta": beta,
				"depth": depth,
				"isPlayerBonus": isPlayerBonus,
				"isOpponentBonus": True
			}
			g, move = self.alphaBeta(alphaBeta_params)

			if self.isTimeLeft():
				return g, move
			else:
				if g >= beta:
					lowerBound = g
				else:
					upperBound = g

		return g, move

	def zobristVal(self, formattedBoard, formattedBlockStatus, root):
		hashval = 0
		for i in xrange(0, 16):
			for j in xrange(0, 16):
				tp2 = self.zobrist[i][j][1 + root[0]][1 + root[1]]
				e, f, g, h = i/4, j/4, i%4, j%4
				tp = formattedBoard[e][f][g][h]
				if tp == 2:
					hashval ^= tp2[1]
				elif tp == 1:
					hashval ^= tp2[0]
		return hashval

	def alphaBeta(self, alphaBeta_params):

		board_md5 = self.zobristVal(alphaBeta_params["formattedBoard"], alphaBeta_params["formattedBlockStatus"], alphaBeta_params["root"])

		lower = (-2*self.INFINITY, alphaBeta_params["root"])		
		nLower = 1 + (board_md5 + 1) * (board_md5 + 2)/2

		upper = (2*self.INFINITY, alphaBeta_params["root"])
		nUpper = 2 + (board_md5 + 2) * (board_md5 + 3)/2
		
		if (nLower in self.transpositionTable):
			lower = self.transpositionTable[nLower]
			if (lower[0] >= alphaBeta_params["beta"]):
				return lower
		
		if (nUpper in self.transpositionTable):
			upper = self.transpositionTable[nUpper]
			if (upper[0] <= alphaBeta_params["alpha"]):
				return upper


		tp, stat = self.isTerminal(alphaBeta_params["formattedBoard"], alphaBeta_params["formattedBlockStatus"])
		if tp == self.INFINITY:
			return self.INFINITY, alphaBeta_params["root"]
		elif tp == -self.INFINITY:
			return -self.INFINITY, alphaBeta_params["root"]

		tempBoard = copy.deepcopy(alphaBeta_params["formattedBoard"])
		tempBlockStatus = copy.deepcopy(alphaBeta_params["formattedBlockStatus"])

		alphaBeta_params["alpha"], beta = max(alphaBeta_params["alpha"], lower[0]), min(alphaBeta_params["beta"], upper[0])

		moveHash = self.index + (board_md5 + self.index) * (board_md5 + self.index + 1)/2
		moveInfo = []

		if moveHash not in self.moveOrder:
			children = self.allChildNodes(alphaBeta_params["formattedBoard"], alphaBeta_params["formattedBlockStatus"], alphaBeta_params["root"])
		else:
			children = []
			children += self.moveOrder[moveHash]

		nSiblings = len(children)

		if alphaBeta_params["depth"] == 0 or nSiblings == 0:
			answer = alphaBeta_params["root"]
			g = self.heuristicVal(alphaBeta_params["formattedBoard"], alphaBeta_params["formattedBlockStatus"])

		elif self.myMove:
			g = -self.INFINITY
			answer = children[0]

			# print "AlphaBeta: Before calling child: ", datetime.datetime.utcnow() - self.begin
			if self.isTimeLeft():
				return g, answer
			# print "AlphaBeta: After calling child: ", datetime.datetime.utcnow() - self.begin

			a = alphaBeta_params["alpha"]
			i = 0
			while ((g <alphaBeta_params["beta"]) and (i < nSiblings)):
				self.myMove = False
				c = children[i]
				e = c[0]/4
				f = c[1]/4
				w = c[0]%4
				h = c[1]%4
				tempBoard[e][f][w][h] = 1
				tempBlockStatus[e][f] = self.getBlockStatus(tempBoard[e][f])

				self.index += 1

				alphaBeta_params_child = {
					"formattedBoard": tempBoard,
					"formattedBlockStatus": tempBlockStatus,
					"root": c,
					"alpha": a,
					"beta": alphaBeta_params["beta"],
					"depth": alphaBeta_params["depth"] - 1,
					"isPlayerBonus": False,
					"isOpponentBonus": alphaBeta_params["isOpponentBonus"]
				}

				if alphaBeta_params["isPlayerBonus"] and tempBlockStatus[e][f] == 1:					
					self.myMove = True
					val, temp = self.alphaBeta(alphaBeta_params_child)
				else:
					alphaBeta_params_child["isPlayerBonus"] = alphaBeta_params["isPlayerBonus"]
					val, temp = self.alphaBeta(alphaBeta_params_child)


				self.index -= 1
				e = c[0]/4
				f = c[1]/4
				w = c[0]%4
				h = c[1]%4
				tempBoard[e][f][w][h] = 0
				tempBlockStatus[e][f] = self.getBlockStatus(tempBoard[e][f])

				temp = (val, c)
				moveInfo.append(temp)
				i += 1

				if val > g:
					answer = c
					g = val

				a = max(a, g)

			self.myMove = True

		else:
			g = self.INFINITY
			answer = children[0]
			if self.isTimeLeft():
				return g, answer		

			b = alphaBeta_params["beta"]
			i = 0
			while ((g > alphaBeta_params["alpha"]) and (i < nSiblings)):
				self.myMove = True
				c = children[i]
				e = c[0]/4
				f = c[1]/4
				w = c[0]%4
				h = c[1]%4
				tempBoard[e][f][w][h] = 2
				tempBlockStatus[e][f] = self.getBlockStatus(tempBoard[e][f])
				
				self.index += 1

				alphaBeta_params_child = {
					"formattedBoard": tempBoard,
					"formattedBlockStatus": tempBlockStatus,
					"root": c,
					"alpha": alphaBeta_params["alpha"],
					"beta": b,
					"depth": alphaBeta_params["depth"] - 1,
					"isPlayerBonus": alphaBeta_params["isPlayerBonus"],
					"isOpponentBonus": False
				}

				if alphaBeta_params["isOpponentBonus"] and tempBlockStatus[e][f] == 2:
					self.myMove = False
					val, temp = self.alphaBeta(alphaBeta_params_child)
				else:
					alphaBeta_params_child["isOpponentBonus"] = alphaBeta_params["isOpponentBonus"]
					val, temp = self.alphaBeta(alphaBeta_params_child)

				e = c[0]/4
				f = c[1]/4
				w = c[0]%4
				h = c[1]%4
				tempBoard[e][f][w][h] = 0
				tempBlockStatus[e][f] = self.getBlockStatus(tempBoard[e][f])
				self.index -= 1

				temp = (val, c)
				moveInfo.append(temp)
				i += 1

				if val < g:
					answer = c
					g = val

				b = min(b, g)
			self.myMove = False

		temp = []

		moveInfo = sorted(moveInfo, reverse = True) if self.myMove else sorted(moveInfo)

		for i in moveInfo:
			children.remove(i[1])
			temp.append(i[1])


		self.moveOrder[moveHash] = []
		self.moveOrder[moveHash] += temp

		random.shuffle(children)
		self.moveOrder[moveHash] += children

		if g <= alphaBeta_params["alpha"]:
			self.transpositionTable[nUpper] = g, answer

		if g >= alphaBeta_params["beta"]:
			self.transpositionTable[nLower] = g, answer

		return g, answer


	def checkValidBlocks(self, prevMove, BlockStatus):

		if prevMove[0] < 0 or prevMove[1] < 0:
			return self.allList

		if BlockStatus[prevMove[0] % 4][prevMove[1] % 4] == 0:
			return ((prevMove[0] % 4, prevMove[1] % 4), )

		return tuple(i for i in self.allList if BlockStatus[i[0]][i[1]] == 0)


	def getBlockStatus(self, block):
		# 1->we won, 0->nothing, 3->drawn, 2->other won

		if block[1][2] == block[3][2] and block[2][1] == block[1][2] and block[3][2] == block[2][3]:
			if block[2][1] == 1:
				return 1
			elif block[2][1] == 2:
				return 2
		if block[1][1] == block[3][1] and block[2][0] == block[1][1] and block[3][1] == block[2][2]:
			if block[2][0] == 1:
				return 1
			elif block[2][0] == 2:
				return 2
		if block[0][2] == block[2][2] and block[1][1] == block[0][2] and block[2][2] == block[1][3]:
			if block[1][1] == 1:
				return 1
			elif block[1][1] == 2:
				return 2
		if block[0][1] == block[2][1] and block[1][0] == block[0][1] and block[2][1] == block[1][2]:
			if block[1][0] == 1:
				return 1
			elif block[1][0] == 2:
				return 2

		for i in xrange(4):
			if block[1][i] == block[2][i] and block[0][i] == block[1][i] and block[2][i] == block[3][i]:
				if block[0][i] == 1:
					return 1
				elif block[0][i] == 2:
					return 2
			if block[i][1] == block[i][2] and block[i][0] == block[i][1] and block[i][2] == block[i][3]:
				if block[i][0] == 1:
					return 1
				elif block[i][0] == 2:
					return 2

		tp = [(i, j) for i in xrange(4) for j in xrange(4) if block[i][j] == 0]

		if not len(tp):
			return 3

		return 0



	def getGameStatus(self, board, oppbBoard, blockProb, revBlockProb):

		new_board = copy.deepcopy(board)
		new_oppboard = copy.deepcopy(oppbBoard)

		bs = [[self.getBlockStatus(new_board[j][i]) for i in xrange(4)] for j in xrange(4)]
		obs = [[self.getBlockStatus(new_oppboard[j][i]) for i in xrange(4)] for j in xrange(4)]

		isOneWin = [[0] * 4 for i in xrange(0, 4)]
		countTwoWin = [[0] * 4 for i in xrange(0, 4)]
		countThreeWin = [[0] * 4 for i in xrange(0, 4)]

		ret = 0

		for i in xrange(4):
			flag = 1
			for j in xrange(4):
				if bs[i][j] == 2 or bs[i][j] == 3:
					flag = 0
			if flag:
				tp = 0
				for j in xrange(4):
					if bs[i][j] == 1:
						tp += 1
				if tp == 3:
					for j in xrange(4):
						isOneWin[i][j] = 1
				elif tp == 2:
					for j in xrange(4):
						countTwoWin[i][j] += 1
				elif tp == 1:
					for j in xrange(4):
						countThreeWin[i][j] += 1


		for j in xrange(4):
			flag = 1
			for i in xrange(4):
				if bs[i][j] == 2 or bs[i][j] == 3:
					flag = 0
			if flag:
				tp = 0
				for i in xrange(4):
					if bs[i][j] == 1:
						tp += 1
				if tp == 3:
					for i in xrange(4):
						isOneWin[i][j] = 1
				elif tp == 2:
					for i in xrange(4):
						countTwoWin[i][j] += 1
				elif tp == 1:
					for i in xrange(4):
						countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond1:
			if bs[i][j] == 2 or bs[i][j] == 3:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond1:
				if bs[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond1:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond1:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond1:
					countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond2:
			if bs[i][j] == 2 or bs[i][j] == 3:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond2:
				if bs[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond2:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond2:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond2:
					countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond3:
			if bs[i][j] == 2 or bs[i][j] == 3:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond3:
				if bs[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond3:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond3:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond3:
					countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond4:
			if bs[i][j] == 2 or bs[i][j] == 3:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond4:
				if bs[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond4:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond4:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond4:
					countThreeWin[i][j] += 1

		for i in xrange(4):
			for j in xrange(4):
				if isOneWin[i][j] == 1:
					ret += self.one_win_board
				else:
					ret += countTwoWin[i][j]*countTwoWin[i][j]*self.two_win_board + countThreeWin[i][j]*countThreeWin[i][j]*self.three_win_board

		isOneWin = [[0] * 4 for i in xrange(0, 4)]
		countTwoWin = [[0] * 4 for i in xrange(0, 4)]
		countThreeWin = [[0] * 4 for i in xrange(0, 4)]

		for i in xrange(4):
			flag = 1
			for j in xrange(4):
				if obs[i][j] == 2 or obs[i][j] == 3:
					flag = 0
			if flag:
				tp = 0
				for j in xrange(4):
					if obs[i][j] == 1:
						tp += 1
				if tp == 3:
					for j in xrange(4):
						isOneWin[i][j] = 1
				elif tp == 2:
					for j in xrange(4):
						countTwoWin[i][j] += 1
				elif tp == 1:
					for j in xrange(4):
						countThreeWin[i][j] += 1


		for j in xrange(4):
			flag = 1
			for i in xrange(4):
				if obs[i][j] == 2 or obs[i][j] == 3:
					flag = 0
			if flag:
				tp = 0
				for i in xrange(4):
					if obs[i][j] == 1:
						tp += 1
				if tp == 3:
					for i in xrange(4):
						isOneWin[i][j] = 1
				elif tp == 2:
					for i in xrange(4):
						countTwoWin[i][j] += 1
				elif tp == 1:
					for i in xrange(4):
						countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond1:
			if obs[i][j] == 2 or obs[i][j] == 3:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond1:
				if obs[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond1:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond1:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond1:
					countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond2:
			if obs[i][j] == 2 or obs[i][j] == 3:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond2:
				if obs[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond2:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond2:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond2:
					countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond3:
			if obs[i][j] == 2 or obs[i][j] == 3:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond3:
				if obs[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond3:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond3:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond3:
					countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond4:
			if obs[i][j] == 2 or obs[i][j] == 3:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond4:
				if obs[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond4:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond4:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond4:
					countThreeWin[i][j] += 1

		for i in xrange(4):
			for j in xrange(4):
				if isOneWin[i][j] == 1:
					ret -= self.one_win_board
				else:
					ret -= countTwoWin[i][j]*countTwoWin[i][j]*self.two_win_board + countThreeWin[i][j]*countThreeWin[i][j]*self.three_win_board
		return ret

	def getCellVal(self, block):

		new_block = copy.deepcopy(block)
		isOneWin = [[0] * 4 for i in xrange(0, 4)]
		countTwoWin = [[0] * 4 for i in xrange(0, 4)]
		countThreeWin = [[0] * 4 for i in xrange(0, 4)]

		for i in xrange(4):
			flag = 1
			for j in xrange(4):
				if new_block[i][j] == 2:
					flag = 0
			if flag:
				tp = 0
				for j in xrange(4):
					if new_block[i][j] == 1:
						tp += 1
				if tp == 3:
					for j in xrange(4):
						isOneWin[i][j] = 1
				elif tp == 2:
					for j in xrange(4):
						countTwoWin[i][j] += 1
				elif tp == 1:
					for j in xrange(4):
						countThreeWin[i][j] += 1


		for j in xrange(4):
			flag = 1
			for i in xrange(4):
				if new_block[i][j] == 2:
					flag = 0
			if flag:
				tp = 0
				for i in xrange(4):
					if new_block[i][j] == 1:
						tp += 1
				if tp == 3:
					for i in xrange(4):
						isOneWin[i][j] = 1
				elif tp == 2:
					for i in xrange(4):
						countTwoWin[i][j] += 1
				elif tp == 1:
					for i in xrange(4):
						countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond1:
			if new_block[i][j] == 2:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond1:
				if new_block[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond1:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond1:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond1:
					countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond2:
			if new_block[i][j] == 2:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond2:
				if new_block[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond2:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond2:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond2:
					countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond3:
			if new_block[i][j] == 2:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond3:
				if new_block[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond3:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond3:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond3:
					countThreeWin[i][j] += 1

		flag = 1
		for (i, j) in self.diamond4:
			if new_block[i][j] == 2:
				flag = 0
		if flag:
			tp = 0
			for (i, j) in self.diamond4:
				if new_block[i][j] == 1:
					tp += 1
			if tp == 3:
				for (i, j) in self.diamond4:
					isOneWin[i][j] = 1
			elif tp == 2:
				for (i, j) in self.diamond4:
					countTwoWin[i][j] += 1
			elif tp == 1:
				for (i, j) in self.diamond4:
					countThreeWin[i][j] += 1

		ret = 0
		for i in xrange(4):
			for j in xrange(4):
				if isOneWin[i][j] == 1:
					ret += self.one_win
				else:
					ret += countTwoWin[i][j]*countTwoWin[i][j]*self.two_win + countThreeWin[i][j]*countThreeWin[i][j]*self.three_win
		return ret

	def getBlockVal(self, block):
		block = tuple([ tuple(block[i]) for i in xrange(4) ])
		if block in self.heuristicDict:
			return self.heuristicDict[block]
		else:
			blockStat = self.getBlockStatus(block)
			if blockStat == 1:
				self.heuristicDict[block] = self.block_won
				return self.block_won
			elif blockStat == 2 or blockStat == 3:
				self.heuristicDict[block] = 0
				return 0
			else:
				self.heuristicDict[block] = self.getCellVal(block)
				return self.heuristicDict[block]


	def heuristicVal(self, formattedBoard, formattedBlockStatus):
		tp, stat = self.isTerminal(formattedBoard, formattedBlockStatus)
		if stat:
			return tp

		blockProb = [ [0] * 4 for i in xrange(0, 4) ]
		
		revCurrenBoard = copy.deepcopy(formattedBoard)
		revBlockProb = [ [0] * 4 for i in xrange(0, 4) ]

		ret = 0
		for r in xrange(0, 4):
			for c in xrange(0, 4):
				blockProb[r][c] = self.getBlockVal(formattedBoard[r][c])

		for r in xrange(0, 4):
			for c in xrange(0, 4):

				for i in xrange(0, 4):
					for j in xrange(0, 4):
						if formattedBoard[r][c][i][j] == 1:
							revCurrenBoard[r][c][i][j] = 2
						elif formattedBoard[r][c][i][j] == 2:
							revCurrenBoard[r][c][i][j] = 1

				revBlockProb[r][c] = self.getBlockVal(revCurrenBoard[r][c])
				ret += blockProb[r][c] - revBlockProb[r][c]

		ret += self.getGameStatus(formattedBoard, revCurrenBoard, blockProb, revBlockProb)

		if ret > self.INFINITY:
			return self.INFINITY
		elif ret < -self.INFINITY:
			return -self.INFINITY
		else:
			return ret

	def isTerminal(self, currentBoard, currentBlockStatus):

		tp = self.getBlockStatus(currentBlockStatus)
		if tp == 1:
			return (self.INFINITY, True)
		elif tp == 2:
			return (-self.INFINITY, True)
		elif tp == 0:
			return (0, False)
		else:
			blockCount = 0
			for i in xrange(0, 4):
				for j in xrange(0, 4):
					if currentBlockStatus[i][j] == 1:
						blockCount += 1
					elif currentBlockStatus[i][j] == 2:
						blockCount -= 1
			return (blockCount, True)
