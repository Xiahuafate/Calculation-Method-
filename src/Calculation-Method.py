#/*****************************************************************************/
#/*                                                                           */
#/* Codename : Calculation-Method.py                                          */
#/*                                                                           */
#/* Created:       2021/11/26 (Xvdongyu)                                      */
#/* Last modified: 2016/12/23 (Xvdongyu                                       */
#/* Version:       1.0.0                                                      */
#/*                                                                           */
#/* Description: Claculation-Method's Code                                    */
#/*                                                                           */
#/* Comments:                                                                 */
#/*                                                                           */
#/*                                                                           */
#/*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom
import os
import time
from numpy.core.fromnumeric import size
import xlwt
import datetime
import random

class settings:
    # This a class for the input.xml' settings
    # the attribute of the class are:
    #          self.solution(str): the methods for this problem. The default value is "Conjugate-gradient-method"
    #          self.nmax(int): The maximum number of iterations calculated by iteration. The default value is 1000
    #          self.criteria(float): Convergence criteria for iterative computation. The default value is 1E-07
    #          self.fitting_order(int): fittting order for least square fitting. The default value is 4
    def __init__(self,setting_tag):
        
        try:
            # get the  methods for this problem
            solution_tag = setting_tag.getElementsByTagName("solution")[0]
            self.solution = solution_tag.firstChild.data
        except:
            self.solution = "Conjugate-gradient-method"
            
        try:
            # get the maximum number of iterations calculated by iteration
            nmax_tag = setting_tag.getElementsByTagName("NMAX")[0]
            self.nmax = nmax_tag.firstChild.data
            self.nmax = int(self.nmax)
        except:
            self.nmax = 1000
            
        try:
            # get the convergence criteria for iterative computation.
            convergence_criteria_tag = setting_tag.getElementsByTagName("Convergence-criteria")[0]
            self.criteria = convergence_criteria_tag.firstChild.data
            self.criteria = float(self.criteria)
        except:
            self.criteria = 1E-07
        
        try:
            # get the Fitting order for Least square fitting method
            fitting_order_tag = setting_tag.getElementsByTagName("Fitting-order")[0]
            self.fitting_order = fitting_order_tag.firstChild.data
            self.fitting_order = int(self.fitting_order)
        except:
            self.fitting_order = 4

        try:
            # get the calculation area of x (use in the nonlinear equation solution)
            calculation_area_x_tag = setting_tag.getElementsByTagName("Calculation-area-x")[0]
            calculation_area_x = calculation_area_x_tag.firstChild.data
            calculation_area_x = calculation_area_x.strip("\n").split()
            self.calculation_area_x = []
            [self.calculation_area_x.append(float(i)) for i in calculation_area_x]
        except:
            self.calculation_area_x = [-1, 1]
            
            
            
class matrix:
    # This is a class for input.xml'matrix. It is a part of data.
    # the attribute of the class are:
    #          self.matrix_tag(dom): the matrix_tag in input.xml.
    #          self.matrix_type(str): The method of how to creat the matrix. The default value is "input".
    #          self.matrix_row(int): the row of matrix. The default value is 1.
    #          self.matrix_column(int): the column of matrix. The default value is 1.
    #          self.matrix_elements(np.matrix): the elements of the matrix. No defalut value.
    # the method of the class is:
    #          getMatrixElements(self): get MatrixElements for self.matrix_elements.
    def __init__(self,matrix_tag):
        
        # get the matrix_tag in input.xml.
        self.matrix_tag = matrix_tag
        
        try:
            # get the method of how to creat the matrix.
            self.matrix_type = matrix_tag.getAttribute("matrix_type")
        except:
            self.matrix_type = "input"
            
        try:
            # get the row of matrix
            self.matrix_row = int(matrix_tag.getAttribute("matrix_row"))
        except:
            self.matrix_row = 1
            
        try:
            # get the column of matrix
            self.matrix_column = int(matrix_tag.getAttribute("matrix_column"))
        except:
            self.matrix_column = 1
            
        try:
            # get the order of the equation(only use in nonlinear system)
            self.order = int(matrix_tag.getAttribute("order"))
        except:
            self.order = 1
    
    def getMatrixElements(self):
        
        matrix_elements_tag = self.matrix_tag.getElementsByTagName("matrix_elements")[0]
        matrix_elements = matrix_elements_tag.firstChild.data
        matrix_elements = matrix_elements.strip("\n").split() # get the matrix element
        index_num = [] # the location that the repeat num
        repeat_num = [] # the times that the repeat num
        num = [] # the value that the repeat num, it is str! str!
        
        # try to find where is the repeat input
        for i in range(len(matrix_elements)):
            if "*" in matrix_elements[i]: # if there is a "*" in the matrix elements ,which can make sure it is a repaet input
                index_num.append(i) # find the location
                repeat_num.append(int(matrix_elements[i][0:int(matrix_elements[i].find("*"))])) # find the repeat times
                num.append((matrix_elements[i][int(matrix_elements[i].find("*"))+1:])) # find the repeat num
            else:
                matrix_elements[i] = float(matrix_elements[i]) # try to change the str input to folat input
                
        if len(num) > 0:
            for i in range(len(num)):
                for j in range(repeat_num[-1-i]): # from the last to the first
                    matrix_elements.insert(index_num[-1-i],float(num[-1-i])) # insert the repaet num, the str is change to folat
                matrix_elements.pop(matrix_elements.index(str(repeat_num[-1-i])+"*"+num[-1-i])) # del the repaet input just like xxx*xxx
                
        self.matrix_elements = np.matrix(np.zeros((self.matrix_row,self.matrix_column))) # create the matrix elements
        
        if self.matrix_type == "input": # if the matrix is not the type matrix 
            for i in range(self.matrix_row):
                for j in range(self.matrix_column):
                    self.matrix_elements[i,j] = matrix_elements[i*self.matrix_column+j]
        elif self.matrix_type == "tridiagonal": # if the matrix is the type tridiagonal matrix
            self.matrix_elements[0,0], self.matrix_elements[0,1] = matrix_elements[1], matrix_elements[2] # the first line
            self.matrix_elements[-1,-2], self.matrix_elements[-1,-1] = matrix_elements[0], matrix_elements[1] # the last line
            for i in range(1,self.matrix_row-1):
                self.matrix_elements[i,i-1], self.matrix_elements[i,i], self.matrix_elements[i,i+1] = matrix_elements[0], matrix_elements[1], matrix_elements[2]
            
    
class data:
    # this is a class for input.xml'data.
    # the attribute of the class are:
    #          self.matrix_A(matrix): the A matrix in input.xml. No defalut value
    #          self.matrix_B(matrix): the B matrix in input.xml. No defalut value
    def __init__(self,data_tag):
        
        try:
            # get the A matrix in input.xml
            matrix_A_tag = data_tag.getElementsByTagName("matrix_A")[0]
            self.matrix_A = matrix(matrix_A_tag)
            self.matrix_A.getMatrixElements()
        except:
            os._exit(0)
            
        try:
            # get the A matrix in input.xml
            matrix_B_tag = data_tag.getElementsByTagName("matrix_B")[0]
            self.matrix_B = matrix(matrix_B_tag)
            self.matrix_B.getMatrixElements()
        except:
            os._exit(0)

def DataPretreatment(filename):
    # this is a function for data pro-processing.
    # the main purpose of this function is try to get the data and setting in the input.xml
    # the input element:
    #          filename(str): the input.xml'path + is's name.
    # the output elements:
    #          matrix_data(data): the data in input.xml
    #          problem_settings(settings): the settings in input.xml
    
    file_xml = minidom.parse(filename) # get the input.xml's tree leavel
    input_tag = file_xml.getElementsByTagName("input")[0] # get the input_tag table
    data_tag = input_tag.getElementsByTagName("data")[0] # get the data_tag table in the input_tag table
    settings_tag = input_tag.getElementsByTagName("settings")[0] # get the settings_tag table in the input_tag table
    matrix_data = data(data_tag) # get the matrix data
    problem_settings = settings(settings_tag) # get the settings of the problem
    return matrix_data, problem_settings # return the matrix_data and problem_settings
    
def DataLog(str1):
    # this is a function for datalog output
    # the input element:
    #          str1: the str that needed to write into the Data.log file.
    now_time = "Now Time Is: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n" # get the current time
    f = open("Data.log","w+") 
    f.write(now_time) # write the current time
    f.write(str1+"\n") # write the str that need to write into the file of data.log
    f.close()
    return 0


def DataXlsWrite(sheet_title,title,data):
    # this is a function to output the xls file
    # the input element:
    #          sheet_title(str): the name of sheet and is also the name of xlsfile that be saved.
    #          title(list(str)): the title of the data
    #          data(list(str)): the data need to be writed into the xlsfile
    workbook = xlwt.Workbook(encoding = "utf-8")
    worksheet = workbook.add_sheet(sheet_title)
    style = xlwt.XFStyle()
    font = xlwt.Font() # Create a font for the style
    font.name = 'Times New Roman' 
    font.bold = True # black body
    font.underline = True # The underline
    font.italic = True # italics
    style.font = font # Set the style
    
    alignment = xlwt.Alignment()
    alignment.horz = 0x02
    alignment.vert = 0x01
    style.alignment = alignment
    
    worksheet.write(0, 0, 'Data-Time', style) # Write data title
    for i in range(len(title)):
        worksheet.write(1, i, title[i], style)
        
    style = xlwt.XFStyle()
    style.num_format_str = 'M/D/YY' # Other options: D-MMM-YY, D-MMM, MMM-YY, h:mm, h:mm:ss, h:mm, h:mm:ss, M/D/YY h:mm, mm:ss, [h]:mm:ss, mm:ss.0
    alignment = xlwt.Alignment()
    alignment.horz = 0x02
    alignment.vert = 0x01
    style.alignment = alignment
    
    worksheet.write(0, 1, datetime.datetime.now(), style) # Marks when the data was created
    
    style = xlwt.XFStyle()
    alignment = xlwt.Alignment()
    alignment.horz = 0x02
    alignment.vert = 0x01
    style.alignment = alignment
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            worksheet.write(2+j, i, str(data[i][j]), style)
    workbook.save(sheet_title + ".xls") # Save the file
    
    return 0


def DataPlot(x,y,title,x_label,y_label):
    # this is a function for data plot.
    # the input element:
    #          x(list): the data in x axis.
    #          y(list): the data in y axis
    #          title(str): the title of the figure and the name of the file saved
    #          x_label(str): the name of x_label
    #          y_label(str): the name of y_label
    plt.axes(yscale = "log") # the y axis is log
    l1 = plt.plot(x, y, color='r',marker='o',linestyle='dashed')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend(handles = [l1], labels = [y_label], loc = "best")
    plt.savefig(title+".jpg") # save the file
    plt.close()
    return 0


def DataAnalysis(solution_name,data):
    # this is a function to analysis the data
    # the input element:
    #          solution_name(str): the name of solution.
    #          data(int/float): the data calculated
    if solution_name == "Conjugate-gradient-method":
        # deal with the data of Conjugate-gradient-method
        sheet_title = "Conjugate-gradient-method"
        x_label = "The number of iterations"
        y_label = "Error"
        title = [x_label,y_label,"Genuine Solution"]
        DataXlsWrite(sheet_title,title,data)
        DataPlot(data[0],data[1],sheet_title,x_label,y_label)
        
    elif solution_name == "Least-square-fitting-method":
        # deal with the data of Least-square-fitting-method
        sheet_title = "Least-square-fitting-method"
        title = ["Root mean square error"]
        DataXlsWrite(sheet_title,title,[[data]])
        
    elif solution_name == "Simple-iteration-method":
        # deal with the data:
        #               the times of iteration
        #               the record of x
        #               the record of x_error
        sheet_title = "Simple-iteration-method"
        title = ["times","k","k-error"]
        DataXlsWrite(sheet_title,title,data)
        
    elif solution_name == "Newton-method":
        sheet_title = "Newton-method"
        x_label = "The number of iterations"
        y_label = "Error"
        title = [x_label,y_label,"the solutions"]
        DataXlsWrite(sheet_title,title,data)
        DataPlot(data[0],data[1],sheet_title,x_label,y_label)
        
    elif solution_name == "Broy-method":
        # deal with the data:
    #               the times of iteration
    #               the record of x_error
    #               the record of solution
        sheet_title = "Broy-method"
        x_label = "The number of iterations"
        y_label = "Error"
        title = [x_label,y_label,"the solutions"]
        DataXlsWrite(sheet_title,title,data)
        DataPlot(data[0],data[1],sheet_title,x_label,y_label)
    return 0
        
        
def ConjugateGradientMethod(matrix_data,nmax,criteria):
    # this is a function to slove the equation by Conjugate Gradient Method.
    # the input element:
    #          matrix_data(data): the data of problem.
    #          nmax(int): the Maximum iteration
    #          criteria(float): Convergence criteria
    # use the fanshu to judge the error
    matrix_A = matrix_data.matrix_A.matrix_elements
    matrix_B = matrix_data.matrix_B.matrix_elements
    ord_num = 2 # the fanshu = 2
    residual = [] # this is the reesidual bewteen the real value and the number value
    x = np.zeros((matrix_B.shape[0],matrix_B.shape[1]))# this is solution of the equation
    # this is the first value for the solution
    solution_error = []
    residual.append(matrix_B-np.dot(matrix_A,x))
    if np.linalg.norm(residual[0],ord_num) < criteria:
        # if the first data is the real data 
        DataLog("The first x is the real x\n")
        DataLog("The first x is 0\n")
        return 0
    d = residual[0]
    for i in range(nmax):
        alpha = np.dot(residual[i].T,d)/np.dot(d.T,np.dot(matrix_A,d))
        x = x + alpha[0,0]*d
        residual.append((matrix_B-np.dot(matrix_A,x)))
        solution_error.append(np.linalg.norm(np.dot(matrix_A,x)-matrix_B,ord_num)/np.linalg.norm(matrix_B,ord_num))
        if solution_error[-1]<criteria:
            break
        else:
            beta = -(np.dot(residual[-1].T,np.dot(matrix_A,d))/np.dot(d.T,np.dot(matrix_A,d)))
            d = residual[-1] + beta[0,0]*d
    if i == (nmax - 1):
        # if the i = the Maximum iteration
        DataLog("The iterative calculation has reached the maximum number of iterations, but the calculation still cannot converge.\n")
    else:
        # deal with the data
        data=[[]]
        data.append(solution_error)
        data.append([])
        for i in range(1,len(solution_error)+1):
            data[0].append(i)
        for i in range(len(x)):
            data[2].append(x[i][0,0])
        DataAnalysis("Conjugate-gradient-method",data)
    return 0
    
    
def LeastSquareFittingMethod(matrix_data, fitting_order):
    # this is a function to slove the equation by Conjugate Gradient Method.
    # the input element:
    #          matrix_data(data): the data of problem.
    #          fitting_order: the order of least square fitting
    x = matrix_data.matrix_A.matrix_elements # the fitting data of x
    y = matrix_data.matrix_B.matrix_elements # the fitting data of y
    size_equation = fitting_order + 1
    
    matrix_A = np.matrix(np.zeros((size_equation, size_equation))) # the Normal equations of this problem
    matrix_B = np.matrix(np.zeros((size_equation, 1))) # the b of this problem
    
    for i in range(size_equation):
    # creat the Normal equations
        for k in range(len(y)):
            matrix_B[i,0] = matrix_B[i,0] + y[k,0] * x[k,0]**i
        for j in range(size_equation):
            for k in range(len(x)):
                matrix_A[i,j] = matrix_A[i,j] + x[k,0]**(i + j)
                
    c = np.linalg.inv(matrix_A)@(matrix_B) # solve the Normal equations
    
    mean_square_error = 0 # the mean square error of the problem
    for i in range(len(x)):
        temp_p = 0 # the temp stroge for p(xi)
        for j in range(size_equation):
            temp_p = temp_p + c[j,0]*x[i,0]**j # p(xi) = a0 + a1*xi + a2*x2*x2 +  
        mean_square_error = mean_square_error + (temp_p - y[i,0])**2 # calculate the mean square error
    mean_square_error = mean_square_error**0.5
    
    DataAnalysis("Least-square-fitting-method",mean_square_error)
    
    return 0 
    
def CalculationPolynomial(coefficient_of_eqution, x):
    # this is a function for Calculation nolinear matrix
    # the input element:
    #          matrix_A(np.matrix): the coefficient matrix 
    #          x(float): the x
    f = 0
    # f = a*x^(n) + b*x(n-1) + ~
    for i in range(1, len(coefficient_of_eqution) + 1):
        f = f + coefficient_of_eqution[-i] * x**(i - 1)
        
    return f

def JacobianMatrix(matrix_A, x, order):
    # this is a function for Jacobian matrix
    # the input element:
    #          matrix_A(np.matrix): the coefficient matrix 
    #          x(np.matrix): solution vector
    #          order: the max oreder of x
    row, column = matrix_A.shape[0], matrix_A.shape[1]
    numbers_of_x = int((column-1)/order) # the numbers of the unknowns
    coefficient = np.zeros([row,numbers_of_x,order]) # the coefficient of the unknowns elements
    # get the coefficient 
    for i in range(row):
        for j in range(numbers_of_x):
            for k in range(order):
                coefficient[i,j,k] = matrix_A[i,(k+j*order)]*(order - k)
    # get the jacobian
    matrix_jacobian = np.zeros([numbers_of_x,numbers_of_x])
    for i in range(numbers_of_x):
        for j in range(numbers_of_x):
            matrix_jacobian[i,j] = CalculationPolynomial(coefficient[i,j],x[j])
    
    return matrix_jacobian
    
def CalculationNoLinearMatrix(matrix_A, x, order):
    # this is a function for Calculation nolinear matrix
    # the input element:
    #          matrix_A(np.matrix): the coefficient matrix 
    #          x(np.matrix): solution vector
    #          order: the max oreder of x
    column = matrix_A.shape[1]
    numbers_of_x = int((column-1)/order)
    matrix_x = np.ones([column,1])
    
    for i in range(numbers_of_x):
        for j in range(order):
            matrix_x[j+i*order] = x[i]**(order-j)
            
    matrix_x = np.matrix(matrix_x)
    result = matrix_A * matrix_x
    
    return result
    
def NewtonMethod(matrix_data,problem_settings):
    # this is a function for Newton method
    # the input element:
    #          matrix_data(data): the data of the problem
    #          problem_settings(settings): the settings of the problem
    nmax = problem_settings.nmax # the nmax times
    criteria = problem_settings.criteria # the criteria 
    order = matrix_data.matrix_A.order # the max order of the x
    matrix_A = matrix_data.matrix_A.matrix_elements # coefficient matrix
    matrix_B = matrix_data.matrix_B.matrix_elements # the matrix b
    ord_num = 2 # norm

    x_first = np.ones([matrix_B.shape[0],matrix_B.shape[1]]) # the k time x
    x_end = np.zeros([matrix_B.shape[0],matrix_B.shape[1]]) # the k-1 time x
    x_error = [] # the error of the calculation
    
    x_error.append(np.linalg.norm(x_first - x_end, ord_num))
    i = 0
    while (x_error[-1]>criteria):
        i= i + 1
        matrix_jacobian = JacobianMatrix(matrix_A, x_first, order)
        matrix_result = -CalculationNoLinearMatrix(matrix_A, x_first, order)
        delt_x = np.linalg.inv(matrix_jacobian)@(matrix_result)
        x_end = x_first
        x_first = x_first + delt_x
        x_error.append(np.linalg.norm(x_first - x_end, ord_num))
            
    if (i == (nmax - 1)):
        DataLog("The number of iterations has reached the upper limit of iterations")
    # deal with the data:
    #               the times of iteration
    #               the record of x_error
    #               the record of solution
    x_data = [[]]
    [x_data[0].append(i + 1) for i in range(len(x_error))]
    x_data.append(x_error)
    x_data.append([])
    [x_data[2].append(x_first[i,0]) for i in range(len(x_first))]
    DataAnalysis("Newton-method",x_data)
    
def BroyMethod(matrix_data, problem_settings):
    # this is a function for Broy method
    # the input element:
    #          matrix_data(data): the data of the problem
    #          problem_settings(settings): the settings of the problem
    nmax = problem_settings.nmax # the nmax times
    criteria = problem_settings.criteria # the criteria 
    order = matrix_data.matrix_A.order # the max order of the x
    matrix_A = matrix_data.matrix_A.matrix_elements # coefficient matrix
    matrix_B = matrix_data.matrix_B.matrix_elements # the matrix b
    ord_num = 2 # norm
    
    x_first = np.ones([matrix_B.shape[0],matrix_B.shape[1]]) # the k time x
    x_end = np.zeros([matrix_B.shape[0],matrix_B.shape[1]]) # the k-1 time x
    x_error = [] # the error of the calculation
    
    x_error.append(np.linalg.norm(x_first - x_end, ord_num))
    matrix_jacobian = JacobianMatrix(matrix_A, x_first, order) # Jacobian matrix
    matrix_result = -CalculationNoLinearMatrix(matrix_A, x_first, order) # f(x_(k-1))
    a = np.linalg.inv(matrix_jacobian) # the matrix A
    delt_x = a@(matrix_result)
    x_end = x_first
    x_first = x_first + delt_x
    x_error.append(np.linalg.norm(x_first - x_end, ord_num))
    i = 0
    while (x_error[-1]>criteria):
        i= i + 1
        s = x_first - x_end # s = x_k - x_(k-1)
        y = CalculationNoLinearMatrix(matrix_A, x_first, order) - CalculationNoLinearMatrix(matrix_A, x_end, order)
        # y = f(x_k) - f(x_(k-1))
        temp = a
        a = temp + (s - temp * y) * (s.T) * temp / (1 + (s.T) * temp * y)
        x_end = x_first
        x_first = x_first - a * CalculationNoLinearMatrix(matrix_A, x_end, order)
        # x_(k+1) = x_k - A*f(x_k)
        x_error.append(np.linalg.norm(x_first - x_end, ord_num))
    if (i == (nmax - 1)):
        DataLog("The number of iterations has reached the upper limit of iterations")
    # deal with the data:
    #               the times of iteration
    #               the record of x_error
    #               the record of solution
    x_data = [[]]
    [x_data[0].append(i + 1) for i in range(len(x_error))]
    x_data.append(x_error)
    x_data.append([])
    [x_data[2].append(x_first[i,0]) for i in range(len(x_first))]
    DataAnalysis("Broy-method",x_data)

def Dichotomy(coefficient_of_equation, calculation_area_x, criteria):
    # this is a function for dichotomy
    # the input element:
    #           coefficient_of_equation(np.matrix): the coefficient of the equation
    #           calculation_area_x(list): the oral area of the calculation
    #           criteria(folat): the length of the need area of the calculation
    # it must need the zero point in the area !
    x_left = calculation_area_x[0] # x point in the left
    x_right = calculation_area_x[-1] # x point in the right
    x_middle = (x_left + x_right) / 2.0 # x point in the middle
    while(1):
        f_left = CalculationPolynomial(coefficient_of_equation, x_left) # the f(x_left)
        f_right = CalculationPolynomial(coefficient_of_equation, x_right) # the f(x_right)
        f_middle = CalculationPolynomial(coefficient_of_equation, x_middle) # the f(x_middle)

        if (f_left * f_middle) <= 0 :
            x_right = x_middle
            x_middle = (x_left + x_right) / 2.0
        elif (f_right * f_middle) <= 0:
            x_left = x_middle
            x_middle = (x_left + x_right) / 2.0
        else:
            os._exit(0)

        if abs(x_right - x_left) < criteria:
            break
    # the need area calculation 
    calculation_area_x[0] = x_left
    calculation_area_x[-1] = x_right

    return calculation_area_x

        

def SimpleIterationMethod(matrix_data, problem_settings):
    # this is a function for Simple Iteration Method
    # the input element:
    #          matrix_data(data): the data of the problem
    #          problem_settings(settings): the settings of the problem
    # this function is used for solve a x equation only about x 
    criteria = problem_settings.criteria # the criteria
    calculation_area_x = problem_settings.calculation_area_x # the calculation area of x
    coefficient_of_equation = matrix_data.matrix_A.matrix_elements # Coefficient of equation
    # firt need to divde the calculation of area
    calculation_area_x = Dichotomy(coefficient_of_equation, calculation_area_x, criteria*10000)
    # then find the zero point
    # get a random point in the area
    x_first = random.uniform(calculation_area_x[0], calculation_area_x[-1]) # the k times result of calculation
    x_end = 0 # the k-1 times result of calculation
    x_record = [x_first] # record every x in the calculation
    x_error = [abs(x_first - x_end)] # record every x_error in the calculation
    coefficient_of_order = coefficient_of_equation[0,0] # the coefficient of the max order of x
    coefficient_of_equation[0] = 0
    while (abs(x_first - x_end) > criteria):
        x_end = x_first
        # get the result of polynomial
        x_first = CalculationPolynomial(coefficient_of_equation, x_end) / (0.0 - coefficient_of_order)
        x_first = x_first[0,0]
        # get the new x
        x_first = pow(x_first, 1.0/6.0) / (coefficient_of_order)
        x_record.append(x_first)
        x_error.append(abs(x_first - x_end))
    # deal with the data:
    #               the times of iteration
    #               the record of x
    #               the record of x_error
    x_data = [[]]
    [x_data[0].append(i + 1) for i in range(len(x_record))]
    x_data.append(x_record)
    x_data.append(x_error)
    DataAnalysis("Simple-iteration-method",x_data)

    return 0


    
def MethodSelect(matrix_data,problem_settings):
    # this is a function for Method select
    # the input element:
    #          matrix_data(data): the data of the problem
    #          problem_settings(settings): the settings of the problem
    # the use is obvious
    if problem_settings.solution == "Conjugate-gradient-method":
        ConjugateGradientMethod(matrix_data,problem_settings.nmax,problem_settings.criteria)
        
    elif problem_settings.solution == "Least-square-fitting-method":
        LeastSquareFittingMethod(matrix_data,problem_settings.fitting_order)
        
    elif problem_settings.solution == "Newton-method":
        NewtonMethod(matrix_data, problem_settings)
        
    elif problem_settings.solution == "Simple-iteration-method":
        SimpleIterationMethod(matrix_data, problem_settings)
        
    elif problem_settings.solution == "Broy-method":
        BroyMethod(matrix_data, problem_settings)

    return 0
        

if __name__ == "__main__":
    # for main!
    # the input file name 
    input_filename = "D:\\Document\\Python\\Calculation-Method-\\input\\Nonlinear-system-equation-method.xml"
    # read the input card
    matrix_data, problem_settings = DataPretreatment(input_filename)
    # select the method and do the calculations
    MethodSelect(matrix_data,problem_settings)
    