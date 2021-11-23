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
            self.solition = "Conjugate-gradient-method"
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
        matrix_elements = matrix_elements.strip("\n").split()
        index_num = []
        repeat_num = []
        num = []
        for i in range(len(matrix_elements)):
            if "*" in matrix_elements[i]:
                index_num.append(i)
                repeat_num.append(int(matrix_elements[i][0:int(matrix_elements[i].find("*"))]))
                num.append(float(matrix_elements[i][int(matrix_elements[i].find("*"))+1:]))
            else:
                matrix_elements[i] = float(matrix_elements[i])
        if len(num) > 0:
            index_insert = 0
            for i in range(len(num)):
                for j in range(repeat_num[i]):
                    matrix_elements.insert(index_num[i]+sum(repeat_num[:i])-i,num[i])
        self.matrix_elements = np.matrix(np.zeros((self.matrix_row,self.matrix_column)))
        if self.matrix_type == "input":
            for i in range(self.matrix_row):
                for j in range(self.matrix_column):
                    self.matrix_elements[i,j] = matrix_elements[i*self.matrix_column+j]
        elif self.matrix_type == "tridiagonal":
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
    




if __name__ == "__main__":
    input_filename = "D:\Document\Python\Calculation-Method-\input\Conjugate-gradient-method-input.xml"
    DataPretreatment(input_filename)