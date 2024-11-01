import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        seq_len = len(y_probs[0])
        for i in range(seq_len):
            index = np.argmax(y_probs[:, i, 0])
            path_prob = path_prob * y_probs[index, i, 0]
            if index == 0:
                blank = 1
            else:
                if blank != 1:
                    if len(decoded_path) == 0 or decoded_path[-1] != self.symbol_set[index - 1]:
                        path = self.symbol_set[index - 1]
                        decoded_path.append(path)
                else:
                    blank = 0
                    path = self.symbol_set[index - 1]
                    decoded_path.append(path)
        decoded_path = "".join(decoded_path)
        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width


    def obtainSymbol(self, x, prob):
        score_ = {}
        symbol = []
        for p in self.path:
            idx = 0
            for sym in self.symbol_set:
                p_ = p + sym
                score_[p_] = prob[idx + 1, x, 0] * self.score[p]
                symbol.append(p_)
                idx += 1
        for sym in self.symbol:
            idx = 0
            for s in self.symbol_set:
                if s != sym[-1]:
                    sym_ = sym + s
                else:
                    sym_ = sym
                if sym_ not in score_:
                    symbol.append(sym_)
                    score_[sym_] = prob[idx + 1, x, 0] * self.score_[sym]
                else:
                    score_[sym_] = score_[sym_] + prob[idx + 1, x, 0] * self.score_[sym]
                idx += 1
        return symbol, score_

    def obtainPath(self, x, prob):
        score = {}
        path = []
        for p in self.path:
            score[p] = prob[0, x, 0] * self.score[p]
            path.append(p)
        for sym in self.symbol:
            if sym not in path:
                path.append(sym)
                score[sym] = prob[0, x, 0] * self.score_[sym]
            else:
                score[sym] =  score[sym] + prob[0, x, 0] * self.score_[sym]

        return path, score

    def select(self):
        scoreContainer = []
        symbol = []
        score_ = {}
        path = []
        score = {}

        for s in self.score.values():
            scoreContainer.append(s)

        for s in self.score_.values():
            scoreContainer.append(s)

        scoreContainer = sorted(scoreContainer)

        if len(scoreContainer) < self.beam_width:
            bar = scoreContainer[-1]
        else:
            bar = scoreContainer[-self.beam_width]

        for p in self.path:
            if self.score[p] >= bar:
                score[p] = self.score[p]
                path.append(p)
        for sym in self.symbol:
            if self.score_[sym] >= bar:
                score_[sym] = self.score_[sym]
                symbol.append(sym)
        return symbol, path, score_, score

    def combine(self):
        path = []
        score = {}

        for p in self.path:
            if p not in path:
                path.append(p)
                score[p] = self.score[p]
            else:
                score[p] += self.score[p]

        for sym in self.symbol:
            if sym not in path:
                path.append(sym)
                score[sym] = self.score_[sym]
            else:
                score[sym] += self.score_[sym]
        new_score = sorted(score.items(), key=lambda k: (k[1], k[0]))
        new_path = new_score[-1][0]
        return new_path, score

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        self.symbol = []
        for s in self.symbol_set:
            self.symbol.append(s)
        self.score_ = {}
        self.path = [""]
        self.score = {"" : y_probs[0, 0, 0]}
        sym_len = len(self.symbol_set)
        for i in range(sym_len):
            self.score_[self.symbol_set[i]] = y_probs[1 + i, 0, 0]

        for i in range(1, T):
            self.symbol, self.path, self.score_, self.score = self.select()
            path, score = self.obtainPath(i, y_probs)
            symbol, score_ = self.obtainSymbol(i, y_probs)
            self.score_ = score_
            self.symbol = symbol
            self.path = path
            self.score = score

        bestPath, FinalPathScore = self.combine()

        return bestPath, FinalPathScore
