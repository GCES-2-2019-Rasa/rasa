# noinspection PyPep8Naming
class DialogueTrainingData:
    def __init__(self, variables, y, true_length=None):
        self.variables = variables
        self.y = y
        self.true_length = true_length

    def limit_training_data_to(self, max_samples):
        self.variables = self.variables[:max_samples]
        self.y = self.y[:max_samples]
        self.true_length = self.true_length[:max_samples]

    def is_empty(self):
        """Check if the training matrix does contain training samples."""
        return self.variables.shape[0] == 0

    def max_history(self):
        return self.variables.shape[1]

    def num_examples(self):
        return len(self.y)

    def shuffled_X_y(self):
        import numpy as np

        idVariables = np.arange(self.num_examples())
        np.random.shuffle(idVariables)
        shuffled_variables = self.variables[idVariables]
        shuffled_y = self.y[idVariables]
        return shuffled_variables, shuffled_y
