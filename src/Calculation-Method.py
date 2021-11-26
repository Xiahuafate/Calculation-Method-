import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom
import os

class settings:
    def __init__(self,setting_tag):
        try:
            solution_tag = setting_tag.getElementsByTagName("solution")[0]
            self.solution = solution_tag.firstChild.data
        except:
            self.solution = "Conjugate-gradient-method"
        try:
            nmax_tag = setting_tag.getElementsByTagName("NMAX")[0]
            self.nmax = nmax_tag.firstChild.data
            self.nmax = int(self.nmax)
        except:
            self.namx = 1000
        try:
            convergence_criteria_tag = setting_tag.getElementsByTagName("Convergence-criteria")[0]
            self.criteria = convergence_criteria_tag.firstChild.data
            self.criteria = float(self.criteria)
        except:
            self.criteria = 1E-07
            
class matrix:
    def __init__(self,matrix_tag):
        self.matrix_tag = matrix_tag
        try:
            self.matrix_type = matrix_tag.getAttribute("matrix_type")
        except:
            self.matrix_type = "input"
        try:
            self.matrix_row = int(matrix_tag.getAttribute("matrix_row"))
        except:
            self.matrix_row = 1
        try:
            self.matrix_column = int(matrix_tag.getAttribute("matrix_column"))
        except:
            self.matrix_column = 1
    
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
            self.matrix_elements[0,0], self.matrix_elements[0,1] = matrix_elements[1], matrix_elements[2]
            self.matrix_elements[-1,-2], self.matrix_elements[-1,-1] = matrix_elements[0], matrix_elements[1]
            for i in range(1,self.matrix_row-1):
                self.matrix_elements[i,i-1], self.matrix_elements[i,i], self.matrix_elements[i,i+1] = matrix_elements[0], matrix_elements[1], matrix_elements[2]
            
    
class data:
    def __init__(self,data_tag):
        try:
            matrix_A_tag = data_tag.getElementsByTagName("matrix_A")[0]
            self.matrix_A = matrix(matrix_A_tag)
            self.matrix_A.getMatrixElements()
        except:
            os._exit(0)
        try:
            matrix_B_tag = data_tag.getElementsByTagName("matrix_B")[0]
            self.matrix_B = matrix(matrix_B_tag)
            self.matrix_B.getMatrixElements()
        except:
            os._exit(0)

def DataPretreatment(filename):
    file_xml = minidom.parse(filename)
    input_tag = file_xml.getElementsByTagName("input")[0]
    data_tag = input_tag.getElementsByTagName("data")[0]
    settings_tag = input_tag.getElementsByTagName("settings")[0]
    matrix_data = data(data_tag)
    problem_settings = settings(settings_tag)
    return matrix_data, problem_settings
    

def ConjugateGradientMethod(matrix_data,namx,criteria):
    # use the fanshu to judge the error
    matrix_A = matrix_data.matrix_A.matrix_elements
    matrix_B = matrix_data.matrix_B.matrix_elements
    ord_num = 2
    residual = [] # this is the reesidual bewteen the real value and the number value
    x = np.zeros((matrix_B.shape[0],matrix_B.shape[1]))# this is solution of the equation
    # this is the first value for the solution
    solution_error = []
    residual.append(matrix_B-np.dot(matrix_A,x))
    if np.linalg.norm(residual[0],ord_num) < criteria:
        return 0
    d = residual[0]
    for i in range(namx):
        alpha = np.dot(residual[i].T,d)/np.dot(d.T,np.dot(matrix_A,d))
        x = x + alpha[0,0]*d
        residual.append((matrix_B-np.dot(matrix_A,x)))
        solution_error.append(np.linalg.norm(np.dot(matrix_A,x)-matrix_B,ord_num)/np.linalg.norm(matrix_B,ord_num))
        if solution_error[-1]<criteria:
            break
        else:
            beta = -(np.dot(residual[-1].T,np.dot(matrix_A,d))/np.dot(d.T,np.dot(matrix_A,d)))
            d = residual[-1] + beta[0,0]*d
    print(1)
    
    
    

def MethodSelect(matrix_data,problem_settings):
    if problem_settings.solution == "Conjugate-gradient-method":
        ConjugateGradientMethod(matrix_data,problem_settings.nmax,problem_settings.criteria)
        


if __name__ == "__main__":
    input_filename = "D:\Document\Python\Calculation-Method-\input\Conjugate-gradient-method-input.xml"
    matrix_data, problem_settings = DataPretreatment(input_filename)
    MethodSelect(matrix_data,problem_settings)
    