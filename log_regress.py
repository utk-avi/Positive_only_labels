import numpy as np
import matplotlib.pyplot as plt
import utility as util  
from linear_mode1 import LinearModel

x_train, y_train = util.load_dataset('ds3_train.csv', add_intercept=True)
_, t_train = util.load_dataset('ds3_train.csv', label_col='t')
x_test, y_test = util.load_dataset('ds3_test.csv', add_intercept=True)
_, t_test = util.load_dataset('ds3_test.csv', label_col='t')



class LogisticRegression(LinearModel):
    
    def fit(self, x, y):
        

        def h(theta, x):
            
            return 1 / (1 + np.exp(-np.dot(x, theta)))

        def gradient(theta, x, y):
           
            m, _ = x.shape
            return -1 / m * np.dot(x.T, (y - h(theta, x)))

        def hessian(theta, x):
            
            m, _ = x.shape
            h_theta_x = np.reshape(h(theta, x), (-1, 1))
            return 1 / m * np.dot(x.T, h_theta_x * (1 - h_theta_x) * x)

        def next_theta(theta, x, y):
            
            return theta - np.dot(np.linalg.inv(hessian(theta, x)), gradient(theta, x, y))

        m, n = x.shape

        # Initialize theta
        if self.theta is None:
            self.theta = np.zeros(n)

        # Update theta using Newton's Method
        old_theta = self.theta
        new_theta = next_theta(self.theta, x, y)
        while np.linalg.norm(new_theta - old_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(old_theta, x, y)

        self.theta = new_theta

    def predict(self, x):

        return x @ self.theta >= 0

log_reg = LogisticRegression()
log_reg.fit(x_train, t_train)

util.plot(x_train, t_train, log_reg.theta)

print("Theta is: ", log_reg.theta)
print("The accuracy on training set is: ", np.mean(log_reg.predict(x_train) == t_train))

plt.show()

