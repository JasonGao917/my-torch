import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """
        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)
        # TODO: initialize skip_connect to be all zeros.
        skip_connect = np.zeros(len(extended_symbols))
        # -------------------------------------------->
        # TODO
        tl = len(target)
        for i in range(1, tl):
            if(target[i - 1] == target[i]):
                pass
            else:
                idx = 1 + 2 * i
                skip_connect[idx] = 1
        # <---------------------------------------------
        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        es0 = extended_symbols[0]
        alpha[0][0] = logits[0, es0]
        # TODO: Intialize alpha[0][1]
        es1 = extended_symbols[1]
        alpha[0][1] = logits[0, es1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        for t in range(1, T):
            lt = logits[t, extended_symbols[0]]
            at_pre = alpha[t - 1, 0]
            alpha[t, 0] = lt * at_pre
            for sym in range(1, S):
                if not skip_connect[sym]:
                    alpha[t, sym] = alpha[t - 1, sym - 1] + alpha[t - 1, sym]
                else:
                    alpha[t, sym] = alpha[t - 1, sym - 2] + alpha[t - 1, sym - 1] + alpha[t - 1, sym]
                lt_cur = logits[t, extended_symbols[sym]]
                alpha[t, sym] = alpha[t, sym] * lt_cur
        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """
        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        beta[-1][-2] = 1
        beta[-1][-1] = 1
        # <--------------------------------------------
        for i in range(T - 2, -1, -1):
            logit_cur = logits[i + 1, extended_symbols[-1]]
            beta_cur = beta[i + 1, -1]
            beta[i, S - 1] = beta_cur * logit_cur
            for j in range(S - 2, -1, -1):
                if j + 3 < S and skip_connect[j + 2]:
                    beta[i, j] = beta[i + 1, j + 2] * logits[i + 1, extended_symbols[j + 2]] + beta[i + 1, j + 1] * logits[i + 1, extended_symbols[j + 1]] + beta[i + 1, j] * logits[i + 1, extended_symbols[j]]
                else:
                    beta[i, j] = beta[i + 1, j + 1] * logits[i + 1, extended_symbols[j + 1]] + beta[i + 1, j] * logits[i + 1, extended_symbols[j]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))
        # -------------------------------------------->
        # TODO
        gamma = beta * alpha
        sumgamma = np.sum(gamma, axis=1)
        gamma /= np.reshape(sumgamma, (-1, 1))
        # <---------------------------------------------
        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

		Initialize instance variables

        Argument(s)
		-----------
		BLANK (int, optional): blank label index. Default 0.

		"""
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

		Computes the CTC Loss by calculating forward, backward, and
		posterior proabilites, and then calculating the avg. loss between
		targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            in_len = input_lengths[batch_itr]
            tar_len = target_lengths[batch_itr]
            logits_ = logits[0:in_len, batch_itr]
            eS, sC = self.ctc.extend_target_with_blank(target[batch_itr, 0:tar_len])
            a = self.ctc.get_forward_probs(logits_, eS, sC)
            b = self.ctc.get_backward_probs(logits_, eS, sC)
            gamma = self.ctc.get_posterior_probs(a, b)
            self.extended_symbols.append(eS)
            self.gammas.append(gamma)
            total_loss[batch_itr] = np.sum(-gamma * np.log(logits_[:, eS]))
            # <---------------------------------------------

        total_loss = np.sum(total_loss) / B

        return total_loss

    def backward(self):
        """

		CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative
		w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            in_len = self.input_lengths[batch_itr]
            l = self.logits[0:in_len, batch_itr]
            eS = self.extended_symbols[batch_itr]
            gamma = self.gammas[batch_itr]
            N = gamma.shape[1]
            for i in range(N):
                dY[0:in_len, batch_itr, eS[i]] = dY[0:in_len, batch_itr, eS[i]] - gamma[:, i]/l[:, eS[i]]
            # <---------------------------------------------
        return dY

