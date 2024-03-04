from matplotlib import pyplot as plt


def print_character_from_name(x):
    for s in x:
        print(s + ' ')


def print_odd_form_to(a, b):
    for x in range(a, b):
        if (x % 2) == 1:
            print(x)


def sum_oddNumber(a, b):
     c=0
     for x in range(a, b):
         if x % 2 == 1:
             c = c + x
     print(c)
def sumNumber_from1_to6():
    c=0
    for x in range(1, 6+1):
       c = c + x
    print(c)


mydict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
def print_key_inMydic():
    for k in mydict.items():
        print(k)


def print_values_inMydic():
    for k,v in mydict.items():
        print(v)

def print_key_and_values():
    for k,v in mydict.items():
        print(f'{k}  ->  {v}')


courses = [131, 141, 142, 212]
names = ['Maths', 'Physics', 'Chem', 'Bio']


def print_num_and_name():
    a = zip(courses, names)
    for x in a:
        print(x)


am = 'jabbawocky'
ab = ['a', 'e', 'o', 'u', 'i']

def find_numberOfConsonants():
    countab = 0
    count = 0
    for y in am:
       for x in ab:
          if y == x:
              countab += 1
       if countab < 1:
            count += 1
       countab = 0

    print(count)


def divideNumber():
    for a in range(-2, 3):
        try:
                print(10/a)
        except ZeroDivisionError:
            print("can’t divided by zero")


ages=[23,10,80]
names=['Hoa','Lam','Nam']


def printSort():
    data = zip(ages,names)
    data = sorted(data, key=lambda x : x[0])
    print(data)

def fileA():
    with open('D:/firstname.txt', 'r') as f:
        for line in f:
            print(line)

def fileACauC():
    with open('D:/firstname.txt', 'r') as f:
        content = f.read()
        print(content)

#define a function

def sum_a_and_b(a, b):
    return a+b


import numpy as np
from scipy.stats import rankdata
M = np.array([[1,2,3],
            [4,5,6],
            [7,8,9]])

V = np.array([1,2,3])

def rankVector(x):
    return np.array(x).argsort().argsort()


def shapeMatrix(x):
    return x.shape

def addMatrixWithLNum(matrix, a):
    matrix = matrix+3


def cau5(x):
    return np.linalg.norm(x)

a=np.array([10,15])
b= np.array([8,2])
c= np.array([1,2,3])
def cau6():
    print(a+b)
    print(a-b)
    # không thể tính print(a-c)

def cau7():
    print(np.dot(a, b))

A=np.array([[2,4,9],[3,6,7]])
def cau8():
    print(np.array(A).argsort().argsort())
    print(a.shape)
    print(tuple(map(list, np.where(A ==7))))
    print(A[:, 1])


def cau9():
    print(np.random.randn(3, 3))

def cau10():
    print(np.identity(3))

def cau11():
    random_matrix = np.random.randint(0, 10, size=(3, 3))
    print(random_matrix)

def cau12():
    print(np.diag([1,2,3]))


def cau13():
    A = np.array([[1, 1, 2], [2, 4, -3], [3, 6, -5]])
    print(np.linalg.det(A))


def cau14():
    a1 = np.array([1, -2, -5])
    a2 = np.array([2, 5, 6])

    h = np.column_stack((a1, a2))
    print(h)


def cau15():
    y_values = np.arange(-5, 6, 1)

    y_squared = y_values **2

    #vẽ đồ thị
    plt.plot(y_values,y_squared, marker='o', linestyle='-')

    plt.title('Bình phương của y')
    plt.xlabel('y')
    plt.ylabel('y^2')

    plt.grid(True)
    plt.show()

def cau16():
    # tạo 4 giá trị cách đều
   a = np.linspace(0, 32, 4)
   print(a)

def cau17():
    x = np.linspace(-5, 5, 50)
    y = x ** 2
    plt.plot(x, y, marker='o', linestyle='-')

    plt.title('Bình phương của y')
    plt.xlabel('y')
    plt.ylabel('y^2')

    plt.grid(True)
    plt.show()

def cau18():
    x = np.linspace(-5, 5, 50)
    y = np.exp(x)
    plt.plot(x, y, marker='o', linestyle='-')

    plt.title('Bình phương của y')
    plt.xlabel('y')
    plt.ylabel('y^2')

    plt.grid(True)
    plt.show()

def cau19():
    x = np.linspace(0.0001, 5, 50)
    y = np.log(x)
    plt.plot(x, y, marker='o', linestyle='-')

    plt.title('Bình phương của y')
    plt.xlabel('y')
    plt.ylabel('y^2')

    plt.grid(True)
    plt.show()

def cau20():


if __name__ == '__main__':
    # print_character_from_name('chuong')
    # print_odd_form_to(1, 10)
    # sum_oddNumber(1, 10)
    # sumNumber_from1_to6()
    # print_key_inMydic()
    # print_values_inMydic()
    # print_key_and_values()
    # print_num_and_name()
    # find_numberOfConsonants()
    # divideNumber()\
    # printSort()
    # fileA()
    # fileACauC()
    # print(sum_a_and_b(3, 5))
    # print(M)
    # print(rankVector(M))
    # print(rankVector(V))
    # chuyen vi ma tran
    # print(np.transpose(M))
    # print(V.T)
    # print(cau5(M))
    # cau6()
    # cau7()
    # cau8()
    # cau9()
    # cau10()
    #cau11()
    # cau12()
    # cau13()
    # cau14()
    cau18()



