import numpy as np
import math


class ITVeALS():

    def __init__(self, ratingMatrix, F, λ, Wui):
        self.ratingMatrix = ratingMatrix
        self.F = F
        self.λ = λ
        self.Wui = Wui

    def __initPQ(self, userSum, itemSum):
        self.Pui = self.ratingMatrix
        self.U = np.zeros((userSum, self.F))
        self.I = np.zeros((itemSum, self.F))
        for i in range(userSum):
            self.U[i] = [np.random.random() / math.sqrt(self.F) for x in range(self.F)]
        for i in range(itemSum):
            self.I[i] = [np.random.random() / math.sqrt(self.F) for x in range(self.F)]

    def iteration_train(self, max_iter):
        userSum = len(self.ratingMatrix)
        itemSum = len(self.ratingMatrix[0])
        self.__initPQ(userSum, itemSum)
        print("ITVeALS training")
        for step in range(max_iter):
            print(f"============Iterations{step}=============")
            for user in range(userSum):
                for f in range(self.F):
                    sum_x = 0.
                    sum_y = 0.
                    for item in range(itemSum):
                        eui = self.Pui[user, item] - self.predict(user, item)
                        sum_x += (eui + self.U[user, f] * self.I[item, f]) * self.I[item, f] * self.Wui[user, item]
                        sum_y += self.I[item, f] ** 2 * self.Wui[user, item]
                    sum_y += self.λ
                    self.U[user, f] = sum_x / sum_y
            for item in range(itemSum):
                for f in range(self.F):
                    sum_x = 0.
                    sum_y = 0.
                    for user in range(userSum):
                        eui = self.Pui[user, item] - self.predict(user, item)
                        sum_x += (eui + self.U[user, f] * self.I[item, f]) * self.U[user, f] * self.Wui[user, item]
                        sum_y += self.U[user, f] ** 2 * self.Wui[user, item]
                    sum_y += self.λ
                    self.I[item, f] = sum_x / sum_y
        result = np.dot(self.U, self.I.T)
        return result

    def predict(self, user, item):
        I_T = self.I.T
        pui = np.dot(self.U[user, :], I_T[:, item])
        return pui
