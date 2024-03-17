import numpy as np

class HammingNetwork:
    def __init__(self, *protos: list[list[int]]) -> None:
        """Expects one or more prototype vectors (arrays) for initialising the network.
        Assumption: all entries in the prototype vectors are either +1 or -1.
        """
        self.w1 	= np.array(protos)
        self.iter   = 1000
        self.b1 	= np.ones((self.w1.shape[0], 1)) * self.w1.shape[1]

        w2_size     = self.w1.shape[0]
        epsilon     = 1 / w2_size
        self.w2     = np.array([[1 if i == j else -epsilon  \
                      for j in range(0, w2_size)]           \
                      for i in range(0, w2_size)])

    def first_layer(self, input: np.ndarray) -> np.ndarray:
        return (self.w1 @ input) + self.b1

    def second_layer(self, input: np.ndarray) -> np.ndarray:
        for i in range(0, self.iter):
            result = self.poslin(self.w2 @ input)
            if (np.array_equal(input, result)):
                break
            input = result
        return result

    def poslin(self, input: np.ndarray) -> np.ndarray:
        return (input > 0) * input

    def classify(self, input: np.ndarray) -> np.ndarray:
        return self.second_layer(self.first_layer(input))

if __name__ == "__main__":
    vectors = list()
    for i in range(0, int(input("Enter the number of parameter vectors: "))):
        vectors.append(list(map(int, input(f"Vector {i}: ").split())))
    model = HammingNetwork(*vectors)

    inp_obj = np.array([[el] for el in map(int, input("Input vector: ").split())])
    answer  = model.classify(inp_obj)
    print("Answer:\n", answer)
