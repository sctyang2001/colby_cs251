'''linear_regression.py
Scottie YANG Miaoyi
CS251 Data Analysis Visualization
Fall 2021
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.stats as stats
import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.m_sse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1
        self.normal = False

    def linear_regression(self, ind_vars, dep_var, p = 1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        N = self.data.get_num_samples()
        self.p = p
        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])

        if p == 1:
            Ahat = np.hstack((np.ones((N,1)),self.A))
        else:
            Ahat = np.hstack((np.ones((N,1)),self.make_polynomial_matrix(self.A, self.p)))
        c, _, _, _ = scipy.linalg.lstsq(Ahat, self.y)

        self.slope = c[1:, :]
        self.intercept = np.squeeze(c[0])

        y_pred = self.predict()

        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)

        self.m_sse = self.mean_sse()

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        if self.p!=1:
            if X is not None:
                y_pred = self.intercept + X@self.slope[0]
            else: 
                y_pred = self.intercept + self.A@self.slope[0]

            for i in range(self.p-1):
                if X is not None:
                    y_pred += (X**(i+2))@self.slope[i+1]
                else:
                    y_pred += (self.A**(i+2))@self.slope[i+1]
        else:
            if X is not None:
                y_pred = self.intercept + X@self.slope
            else: 
                y_pred = self.intercept + self.A@self.slope

        return y_pred


    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        residue = np.sum(self.compute_residuals(y_pred)**2)
        R2 = 1-(residue/(np.sum(((self.y-np.mean(self.y))**2))))
        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        if self.p == 1 and self.normal == False:
            residuals = self.y-y_pred
        else:
            residuals = np.squeeze(self.y)-y_pred

        return residuals

    def mean_sse(self):
        '''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean sum-of-squares error

        Hint: Make use of self.compute_residuals
        '''
        m_sse = np.sum(self.compute_residuals(self.predict())**2)/self.data.get_num_samples()
        return m_sse

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        if self.R2 is not None:
            new_title = title + str(f" - R2 = {self.r_squared(self.predict())}")
        else:
            new_title = title
        the_x, the_y = super().scatter(ind_var, dep_var, new_title)
        if self.p == 1:
            if self.slope is not None:
                line_x = np.linspace(np.squeeze(the_x).min(), np.squeeze(the_x).max(), 100)
                line_y = self.intercept + np.squeeze(self.slope)*line_x
                plt.plot(line_x, line_y, 'r')
        else:
            Ahat = self.make_polynomial_matrix(the_x, self.p)

            # Make x values for line of regression
            xline = np.linspace(np.squeeze(the_x).min(), np.squeeze(the_x).max(), 100)
            # Compute y values for line of regression
            rline = self.intercept + self.slope[0]*xline
            for i in range(self.slope.shape[0]-1):
                rline += self.slope[i+1]*xline**(i+2)

            plt.plot(xline, rline, 'r')


    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        fig, axes = super().pair_plot(data_vars, fig_sz)

        for i in range(len(data_vars)):
            for j in range(len(data_vars)):

                the_x = self.data.select_data([data_vars[j]])
                the_y = self.data.select_data([data_vars[i]])

                self.linear_regression([data_vars[j]], data_vars[i])
                line_x = np.linspace(np.squeeze(the_x).min(), np.squeeze(the_x).max(), 100)
                line_y = self.intercept + np.squeeze(self.slope)*line_x
                axes[i, j].plot(line_x, line_y, 'r')
                axes[i, j].set_title(str(f"R2 = {self.r_squared(self.predict()):0.2f}"))

        if hists_on_diag==True:
            numVars = len(data_vars)
            for k in range(numVars):
                axes[k, k].remove()
                axes[k, k] = fig.add_subplot(numVars, numVars, k*numVars+k+1)
                if k < numVars-1:
                    axes[k, k].set_xticks([])
                else:
                    axes[k, k].set_xlabel(data_vars[k])
                if k > 0:
                    axes[k, k].set_yticks([])
                else:
                    axes[k, k].set_ylabel(data_vars[k])
                hist_x = self.data.select_data([data_vars[k]])
                axes[k, k].hist(hist_x[:,0], bins=10)
                axes[k, k].set_title(f'{data_vars[k]} histogram')

        

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''
        N = A.shape[0]
        Ahat = np.zeros( (N, p) )
        for i in range(p):
            Ahat[:, i] = np.squeeze(A)**(i+1)

        return Ahat


    # def poly_regression(self, ind_var, dep_var, p):
    #     '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
    #     (Week 2)
    #     NOTE: For single linear regression only (one independent variable only)

    #     Parameters:
    #     -----------
    #     ind_var: str. Independent variable entered in the single regression.
    #         Variable names must match those used in the `self.data` object.
    #     dep_var: str. Dependent variable entered into the regression.
    #         Variable name must match one of those used in the `self.data` object.
    #     p: int. Degree of polynomial regression model.
    #          Example: if p=10, then the model should have terms in your regression model for
    #          x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

    #     TODO:
    #     - This method should mirror the structure of self.linear_regression (compute all the same things)
    #     - Differences are:
    #         - You create a matrix based on the independent variable data matrix (self.A) with columns
    #         appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
    #         - You set the instance variable for the polynomial regression degree (self.p)
    #     '''
    #     self.ind_vars = [ind_var]
    #     self.dep_var = dep_var
    #     self.p = p
    #     N = self.data.get_num_samples()
    #     self.A = self.data.select_data([ind_var])
    #     self.y = self.data.select_data([dep_var])

    #     # Make Ahat
    #     Ahat = np.hstack((np.ones((N,1)),self.make_polynomial_matrix(self.A, self.p)))
    #     # Find c
    #     c, _, _, _ = scipy.linalg.lstsq(Ahat, self.y)

    #     self.slope = c[1:, :]
    #     self.intercept = np.squeeze(c[0])

    #     y_pred = self.predict()
    #     print(y_pred, self.y)

    #     self.residuals = self.compute_residuals(y_pred)

    #     self.R2 = self.r_squared(y_pred)

    #     self.m_sse = self.mean_sse()

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor.
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.slope = slope
        self.intercept = intercept
        self.p = p

        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])
        y_pred = self.predict()
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        self.m_sse = self.mean_sse()


    # Attached is my extension:
    def normal_equation(self, ind_vars, dep_var, p = 1):
        ''' Credit:
        https://stackoverflow.com/questions/46586520/normal-equation-implementation-in-python-numpy/46590409
        '''

        self.ind_vars = ind_vars
        self.dep_var = dep_var
        N = self.data.get_num_samples()
        self.p = p
        self.normal = True
        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])

        # contain my minimized values
        theta = []

        # Create a bias_vector to add to my newly created X vector
        bias_vector = np.ones((N, 1))

        # Reshape my original X(m,) vector so that I can manipulate it with my bias_vector; 
        X = np.reshape(self.A, (N, 1))

        # Combine these two vectors together to get a (m, 2) matrix
        X = np.append(bias_vector, X, axis=1)

        # Normal Equation:
        X_transpose = np.transpose(X)

        # Calculating theta
        theta = np.linalg.inv(X_transpose.dot(X))
        theta = theta.dot(X_transpose)
        theta = theta.dot(self.y)
        self.slope = theta[1]
        self.intercept = np.squeeze(theta[0])
        y_pred = self.predict()
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        self.m_sse = self.mean_sse()

    def ci_scatter(self, ind_var, dep_var, title):
        ''' Credit:
        https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
        '''
        if self.R2 is not None:
            new_title = title+str(f" - R2 = {self.r_squared(self.predict()):0.2f}")
        the_x, the_y = super().scatter(ind_var, dep_var, new_title)

        n = self.data.get_num_samples()  # number of samples
        m = 2  # number of parameters
        dof = n - m  # degrees of freedom
        t = stats.t.ppf(0.95, dof) # Students statistic of interval confidence
        residual = self.predict()-self.y
        std_error = (np.sum(residual**2) / dof)**.5   # Standard deviation of the error
        r2 = self.R2

        # mean squared error
        MSE = self.m_sse

        # to plot the adjusted model
        x_line = np.linspace(np.squeeze(the_x).min(), np.squeeze(the_x).max(), 100)
        y_line = np.polyval([np.squeeze(self.slope), self.intercept], x_line)
        x_mean = np.mean(np.squeeze(the_x))

        # confidence interval
        ci = t * std_error * (1/n + (x_line - x_mean)**2 / np.sum((np.squeeze(the_x) - x_mean)**2))**.5
        plt.plot(x_line, y_line, color = 'royalblue')
        plt.fill_between(x_line, y_line + ci, y_line - ci, color = 'skyblue', label = '95% confidence interval', alpha = 0.7)
        plt.legend(bbox_to_anchor=(1, .25), fontsize=12)


