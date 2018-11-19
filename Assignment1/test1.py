from part1 import *
import numpy as np
import sys
'''
    Unit test 1:
    This file includes unit tests for py.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (60 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 


#-------------------------------------------------------------------------
def test_entropy():
    ''' (4.5 points) entropy '''
    y = np.array([0.,0.])
    e = Tree.entropy(y)
    assert np.allclose(e, 0., atol = 1e-3) 

    y = np.array([2.,2.])
    e = Tree.entropy(y)
    assert np.allclose(e, 0., atol = 1e-3) 

    y = np.array([0.,1.])
    e = Tree.entropy(y)
    assert np.allclose(e, 1.0, atol = 1e-3) 

    y = np.array([0.,1.,0.,1.])
    e = Tree.entropy(y)
    assert np.allclose(e, 1.0, atol = 1e-3) 

    y = np.array([2.,5.])
    e = Tree.entropy(y)
    assert np.allclose(e, 1.0, atol = 1e-3) 

    y = np.array([4.,8.,4.,8.])
    e = Tree.entropy(y)
    assert np.allclose(e, 1.0, atol = 1e-3) 

    y = np.array([0.,1.,1.,1.,1.,1.])
    e = Tree.entropy(y)
    assert np.allclose(e, .65, atol = 1e-3) 

    y = np.array(['apple','apple'])
    e = Tree.entropy(y)
    assert np.allclose(e, 0., atol = 1e-3) 

    y = np.array(['orange','apple'])
    e = Tree.entropy(y)
    assert np.allclose(e, 1., atol = 1e-3) 

    y = np.array(['orange','apple','orange','apple'])
    e = Tree.entropy(y)
    assert np.allclose(e, 1., atol = 1e-3) 

    y = np.array(['orange','apple','banana','pineapple'])
    e = Tree.entropy(y)
    assert np.allclose(e, 2., atol = 1e-3) 

#-------------------------------------------------------------------------
def test_conditional_entropy():
    '''(6 points) conditional entropy '''

    y = np.array([0.,0.])
    x = np.array([1.,1.])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array([0.,1.])
    x = np.array([1.,2.])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array([0.,1.,0.,1.])
    x = np.array([1.,4.,1.,4.])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array([0.,1.,0.,1.])
    x = np.array([1.,1.,4.,4.])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 1., atol = 1e-3) 

    y = np.array([0.,0.,1.])
    x = np.array([1.,4.,4.])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 0.666666666667, atol = 1e-3) 

    y = np.array(['apple','apple'])
    x = np.array(['good','good'])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array(['apple','orange'])
    x = np.array(['good','good'])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 1., atol = 1e-3) 

    y = np.array(['apple','orange'])
    x = np.array(['good','bad'])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 0., atol = 1e-3) 

    y = np.array(['apple','orange','pineapple','banana'])
    x = np.array(['a','a','a','a'])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 2., atol = 1e-3) 

    y = np.array(['apple','orange','pineapple','banana'])
    x = np.array(['a','a','b','b'])
    ce = Tree.conditional_entropy(y,x)
    assert np.allclose(ce, 1., atol = 1e-3) 


#-------------------------------------------------------------------------
def test_information_gain():
    '''(3 points) information gain'''

    y = np.array([0.,1.])
    x = np.array([1.,2.])
    g = Tree.information_gain(y,x)
    assert np.allclose(g, 1., atol = 1e-3) 

    y = np.array([0.,0.])
    x = np.array([1.,1.])
    g = Tree.information_gain(y,x)
    assert np.allclose(g, 0., atol = 1e-3) 
  
    y = np.array([0.,1.,0.,1.])
    x = np.array([1.,1.,4.,4.])
    g = Tree.information_gain(y,x)
    assert np.allclose(g, 0., atol = 1e-3)   


    y = np.array([0.,0.,1.])
    x = np.array([1.,4.,4.])
    g = Tree.information_gain(y,x)
    assert np.allclose(g, 0.251629167388, atol = 1e-3) 

    y = np.array(['apple','orange'])
    x = np.array(['good','bad'])
    g = Tree.information_gain(y,x)
    assert np.allclose(g, 1., atol = 1e-3)


    y = np.array(['apple','apple'])
    x = np.array(['good','bad'])
    g = Tree.information_gain(y,x)
    assert np.allclose(g, 0., atol = 1e-3)


#-------------------------------------------------------------------------
def test_best_attribute():
    '''(4.5 points) best attribute'''

    X = np.array([[1.,1.],
                  [1.,2.]])
    Y = np.array([0.,1.])
    i = Tree.best_attribute(X,Y)
    assert i == 1 

    X = np.array([[2.,1.],
                  [1.,1.]])
    Y = np.array([0.,1.])
    i = Tree.best_attribute(X,Y)
    assert i == 0 

    X = np.array([[1.,1.],
                  [2.,2.],
                  [3.,5.]])
    Y = np.array([0.,1.])
    i = Tree.best_attribute(X,Y)
    assert i == 2 


    X = np.array([[1.,1.],
                  [2.,2.],
                  [6.,2.],
                  [3.,5.]])
    Y = np.array([0.,1.])
    i = Tree.best_attribute(X,Y)
    assert i == 2 


    X = np.array([['apple','apple'],
                  ['low'  ,'low'  ],
                  ['good' ,'bad'  ],
                  ['high' ,'high' ]])
    Y = np.array(['good','bad'])
    i = Tree.best_attribute(X,Y)
    assert i == 2 



#-------------------------------------------------------------------------
def test_split():
    '''(4.5 points) split'''

    X = np.array([['apple','orange','pineapple','banana'],
                  ['high','high','low','low'],
                  ['a','b','c','d']])
    Y = np.array(['good','bad','okay','perfect'])
    C = Tree.split(X,Y,1)

    assert type(C) == dict
    assert len(C) == 2 

    assert isinstance(C['high'], Node)
    assert isinstance(C['low'], Node)

    assert C['high'].X.shape == (3,2)
    assert C['high'].Y.shape == (2,)
    assert C['high'].i == None 
    assert C['high'].C == None 
    assert C['high'].isleaf == False 
    assert C['high'].p == None 

    assert C['high'].X[0,0] == 'apple'
    assert C['high'].X[0,1] == 'orange'
    assert C['high'].X[1,0] == 'high'
    assert C['high'].X[1,1] == 'high'
    assert C['high'].X[2,0] == 'a'
    assert C['high'].X[2,1] == 'b'

    assert C['low'].X.shape == (3,2)
    assert C['low'].Y.shape == (2,)
    assert C['low'].i == None 
    assert C['low'].C == None 
    assert C['low'].isleaf == False 
    assert C['low'].p == None 

    assert C['low'].X[0,0] == 'pineapple'
    assert C['low'].X[0,1] == 'banana'
    assert C['low'].X[2,0] == 'c'
    assert C['low'].X[2,1] == 'd'

    C = Tree.split(X,Y,0)

    assert type(C) == dict
    assert len(C) == 4 

    assert isinstance(C['apple'], Node)
    assert isinstance(C['orange'], Node)
    assert isinstance(C['pineapple'], Node)
    assert isinstance(C['banana'], Node)

    assert C['apple'].X.shape == (3,1)
    assert C['apple'].Y.shape == (1,)
    assert C['apple'].i == None 
    assert C['apple'].C == None 
    assert C['apple'].isleaf == False 
    assert C['apple'].p == None 

    assert C['apple'].X[1,0] == 'high'
    assert C['apple'].X[2,0] == 'a'


    assert C['orange'].X.shape == (3,1)
    assert C['orange'].Y.shape == (1,)
    assert C['orange'].i == None 
    assert C['orange'].C == None 
    assert C['orange'].isleaf == False 
    assert C['orange'].p == None 
    assert C['orange'].isleaf == False 
    assert C['orange'].p == None 

    assert C['orange'].X[1,0] == 'high'
    assert C['orange'].X[2,0] == 'b'

    assert C['pineapple'].X.shape == (3,1)
    assert C['pineapple'].Y.shape == (1,)
    assert C['pineapple'].i == None 
    assert C['pineapple'].C == None 
    assert C['pineapple'].isleaf == False 
    assert C['pineapple'].p == None 

    assert C['pineapple'].X[1,0] == 'low'
    assert C['pineapple'].X[2,0] == 'c'

    assert C['banana'].X.shape == (3,1)
    assert C['banana'].Y.shape == (1,)
    assert C['banana'].i == None 
    assert C['banana'].C == None 
    assert C['banana'].isleaf == False 
    assert C['banana'].p == None 

    assert C['banana'].X[1,0] == 'low'
    assert C['banana'].X[2,0] == 'd'

#-------------------------------------------------------------------------
def test_stop1():
    '''(4.5 points) stop1'''

    Y = np.array(['good','bad','okay','perfect'])
    assert Tree.stop1(Y) == False
    Y = np.array(['good','good','good','good'])
    assert Tree.stop1(Y) == True 
    Y = np.array(['good'])
    assert Tree.stop1(Y) == True 


#-------------------------------------------------------------------------
def test_stop2():
    '''(4.5 points) stop2'''
    X = np.array([['apple','orange','pineapple','banana'],
                  ['high','high','low','low'],
                  ['a','b','c','d']])
    assert Tree.stop2(X) == False

    X = np.array([['apple','apple','apple','apple'],
                  ['high','high','low','low'],
                  ['a','b','c','d']])
    assert Tree.stop2(X) == False

    X = np.array([['apple','apple','apple','apple'],
                  ['high','high','high','high'],
                  ['a','b','c','d']])
    assert Tree.stop2(X) == False



    X = np.array([['apple','apple','apple','apple'],
                  ['high','high','high','high'],
                  ['a','a','a','a']])
    assert Tree.stop2(X) == True


    X = np.array([['apple'],
                  ['high'],
                  ['a']])
    assert Tree.stop2(X) == True

#-------------------------------------------------------------------------
def test_most_common():
    '''(3 points) most_common'''

    Y = np.array(['good','bad','good','perfect'])
    assert Tree.most_common(Y) == 'good'

    Y = np.array(['a','b','b','b','c','c'])
    assert Tree.most_common(Y) == 'b'


#-------------------------------------------------------------------------
def test_build_tree():
    '''(3 points) build_tree'''
    X = np.array([['apple'],
                  ['high'],
                  ['a']])
    Y = np.array(['bad'])
    t = Node(X=X, Y=Y) # root node
    
    # build tree
    Tree.build_tree(t)

    assert t.isleaf == True
    assert t.p == 'bad' 
    assert t.C == None
    assert t.i == None
   
    #------------
    X = np.array([['apple','orange','pineapple','banana'],
                  ['high','high','low','low'],
                  ['a','b','c','d']])
    Y = np.array(['good','good','good','good'])
    t = Node(X=X, Y=Y) # root node
    
    # build tree
    Tree.build_tree(t)

    assert t.isleaf == True
    assert t.p == 'good' 
    assert t.C == None
    assert t.i == None
 
    #------------
    X = np.array([['apple','apple','apple','apple'],
                  ['high','high','high','high'],
                  ['a','a','a','a']])
    Y = np.array(['good','bad','bad','bad'])
    t = Node(X=X, Y=Y) # root node
    
    # build tree
    Tree.build_tree(t)

    assert t.isleaf == True
    assert t.p == 'bad' 
    assert t.C == None
    assert t.i == None

    #------------
    X = np.array([['apple','apple','apple','apple'],
                  ['high','high','high','high'],
                  ['a','a','a','a']])
    Y = np.array(['good','bad','good','good'])

    # root node
    t = Node(X=X, Y=Y)
    
    # build tree
    Tree.build_tree(t)

    assert t.isleaf == True
    assert t.p == 'good' 
    assert t.C == None
    assert t.i == None

    ##------------
    X = np.array([['apple','apple','apple'],
                  ['high','high','high'],
                  ['a','b','a']])
    Y = np.array(['good','bad','good'])
    t = Node(X=X, Y=Y) # root node
    Tree.build_tree(t) # build tree

    assert t.i == 2 
    assert t.isleaf == False 
    assert t.p == 'good' 
    assert type(t.C) == dict
    assert len(t.C) == 2 # two children nodes

    c1 = t.C['a']
    c2 = t.C['b']

    assert isinstance(c1, Node)
    assert c1.isleaf == True
    assert c1.p == 'good' 
    assert c1.C == None 
    assert c1.X.shape == (3,2) 
    assert c1.Y.shape == (2,) 
    assert c1.X[0,0] == 'apple'
    assert c1.X[1,0] == 'high'
    assert c1.X[0,1] == 'apple'
    assert c1.X[1,1] == 'high'
    assert c1.Y[0] == 'good'
    assert c1.Y[1] == 'good'

    assert isinstance(c2, Node)
    assert c2.isleaf == True
    assert c2.p == 'bad' 
    assert c2.C == None 
    assert c2.X.shape == (3,1) 
    assert c2.Y.shape == (1,) 
    assert c2.X[0,0] == 'apple'
    assert c2.X[1,0] == 'high'
    assert c2.Y[0] == 'bad'

    #------------
    X = np.array([['apple','orange','pineapple','banana'],
                  ['high','high','low','low'],
                  ['a','b','c','a']])
    Y = np.array(['good','okay','perfect','bad'])
    t = Node(X=X, Y=Y) # root node
    Tree.build_tree(t) # build tree

    assert t.i == 0 
    assert t.isleaf == False 
    assert t.p == 'good' or t.p == 'okay' or t.p == 'perfect' or t.p == 'bad'
    assert type(t.C) == dict
    assert len(t.C) == 4


    #------------
    X = np.array([['apple','apple','apple','banana'],
                  ['high','low','low','high'],
                  ['a','b','c','a']])
    Y = np.array(['good','okay','perfect','bad'])
    t = Node(X=X, Y=Y) # root node
    Tree.build_tree(t) # build tree

    assert t.i == 2 
    assert t.isleaf == False 
    assert t.p == 'good' or t.p == 'okay' or t.p == 'perfect' or t.p == 'bad'
    assert type(t.C) == dict
    assert len(t.C) == 3 

    c1 = t.C['a']
    c2 = t.C['b']
    c3 = t.C['c']

    assert isinstance(c2, Node)
    assert c2.isleaf == True 
    assert c2.p =='okay' 
    assert c2.X.shape == (3,1) 
    assert c2.Y.shape == (1,) 
    assert c2.X[0,0] == 'apple'
    assert c2.X[1,0] == 'low'
    assert c2.X[2,0] == 'b'
    assert c2.Y[0] == 'okay'

    assert isinstance(c3, Node)
    assert c3.isleaf == True 
    assert c3.p == 'perfect' 
    assert c3.X.shape == (3,1) 
    assert c3.Y.shape == (1,) 
    assert c3.X[0,0] == 'apple'
    assert c3.X[1,0] == 'low'
    assert c3.X[2,0] == 'c'
    assert c3.Y[0] == 'perfect'

    assert isinstance(c1, Node)
    assert c1.isleaf == False 
    assert c1.p == 'good' or c1.p == 'bad'
    assert c1.X.shape == (3,2) 
    assert c1.Y.shape == (2,) 
    assert c1.X[0,0] == 'apple'
    assert c1.X[1,0] == 'high'
    assert c1.X[0,1] == 'banana'
    assert c1.X[1,1] == 'high'
    assert c1.Y[0] == 'good'
    assert c1.Y[1] == 'bad'

    assert type(c1.C) == dict
    assert len(c1.C) == 2 # two children nodes
    assert c1.i == 0 
    
    c1c1 = c1.C['apple']
    c1c2 = c1.C['banana']
    assert isinstance(c1c1, Node)
    assert isinstance(c1c2, Node)
    
    assert c1c1.isleaf == True 
    assert c1c1.p == 'good' 
    assert c1c1.X.shape == (3,1) 
    assert c1c1.Y.shape == (1,) 
    assert c1c1.X[0,0] == 'apple'
    assert c1c1.X[1,0] == 'high'
    assert c1c1.X[2,0] == 'a'
    assert c1c1.Y[0] == 'good'

    assert c1c2.isleaf == True 
    assert c1c2.p == 'bad' 
    assert c1c2.X.shape == (3,1) 
    assert c1c2.Y.shape == (1,) 
    assert c1c2.X[0,0] == 'banana'
    assert c1c2.X[1,0] == 'high'
    assert c1c2.X[2,0] == 'a'
    assert c1c2.Y[0] == 'bad'
    

#-------------------------------------------------------------------------
def test_train():
    '''(4.5 points) train'''

    X = np.array([['apple','apple','apple','banana'],
                  ['high','low','low','high'],
                  ['a','b','c','a']])
    Y = np.array(['good','okay','perfect','bad'])
    t = Tree.train(X,Y) 

    assert t.i == 2 
    assert t.isleaf == False 
    assert t.p == 'good' or t.p == 'okay' or t.p == 'perfect' or t.p == 'bad'
    assert type(t.C) == dict
    assert len(t.C) == 3 

    c1 = t.C['a']
    c2 = t.C['b']
    c3 = t.C['c']

    assert isinstance(c2, Node)
    assert c2.isleaf == True 
    assert c2.p =='okay' 
    assert c2.X.shape == (3,1) 
    assert c2.Y.shape == (1,) 
    assert c2.X[0,0] == 'apple'
    assert c2.X[1,0] == 'low'
    assert c2.X[2,0] == 'b'
    assert c2.Y[0] == 'okay'

    assert isinstance(c3, Node)
    assert c3.isleaf == True 
    assert c3.p == 'perfect' 
    assert c3.X.shape == (3,1) 
    assert c3.Y.shape == (1,) 
    assert c3.X[0,0] == 'apple'
    assert c3.X[1,0] == 'low'
    assert c3.X[2,0] == 'c'
    assert c3.Y[0] == 'perfect'

    assert isinstance(c1, Node)
    assert c1.isleaf == False 
    assert c1.p == 'good' or c1.p == 'bad'
    assert c1.X.shape == (3,2) 
    assert c1.Y.shape == (2,) 
    assert c1.X[0,0] == 'apple'
    assert c1.X[1,0] == 'high'
    assert c1.X[0,1] == 'banana'
    assert c1.X[1,1] == 'high'
    assert c1.Y[0] == 'good'
    assert c1.Y[1] == 'bad'

    assert type(c1.C) == dict
    assert len(c1.C) == 2 # two children nodes
    assert c1.i == 0 
    
    c1c1 = c1.C['apple']
    c1c2 = c1.C['banana']
    assert isinstance(c1c1, Node)
    assert isinstance(c1c2, Node)
    
    assert c1c1.isleaf == True 
    assert c1c1.p == 'good' 
    assert c1c1.X.shape == (3,1) 
    assert c1c1.Y.shape == (1,) 
    assert c1c1.X[0,0] == 'apple'
    assert c1c1.X[1,0] == 'high'
    assert c1c1.X[2,0] == 'a'
    assert c1c1.Y[0] == 'good'

    assert c1c2.isleaf == True 
    assert c1c2.p == 'bad' 
    assert c1c2.X.shape == (3,1) 
    assert c1c2.Y.shape == (1,) 
    assert c1c2.X[0,0] == 'banana'
    assert c1c2.X[1,0] == 'high'
    assert c1c2.X[2,0] == 'a'
    assert c1c2.Y[0] == 'bad'

#-------------------------------------------------------------------------
def test_inference():
    '''(4.5 points) inference'''

    t = Node(None,None) 
    t.isleaf = True
    t.p = 'good job' 

    x = np.array(['apple','high','good','b'])

    y = Tree.inference(t,x)
    assert y == 'good job' 

    #----------------- 
    t = Node(None,None) 
    t.isleaf = False 
    t.i = 1
    c1 = Node(None,None)
    c2 = Node(None,None)
    c1.isleaf= True
    c2.isleaf= True
    
    c1.p = 'c1' 
    c2.p = 'c2' 
    t.C = {'high':c1, 'low':c2}

    x = np.array(['apple','high','good','b'])
    y = Tree.inference(t,x)
    assert y == 'c1' 

    x = np.array(['apple','low','good','b'])
    y = Tree.inference(t,x)
    assert y == 'c2' 


    #----------------- 
    t.i = 2
    t.C = {'good':c1, 'bad':c2}
  
    x = np.array(['apple','low','good','b'])
    y = Tree.inference(t,x)
    assert y == 'c1' 
  
    x = np.array(['apple','low','bad','b'])
    y = Tree.inference(t,x)
    assert y == 'c2' 

    t.p = 'c3'
    x = np.array(['apple','low','okay','b'])
    y = Tree.inference(t,x)
    assert y == 'c3' 




#-------------------------------------------------------------------------
def test_predict():
    '''(4.5 points) predict'''
    t = Node(None,None) 
    t.isleaf = False 
    t.i = 1
    c1 = Node(None,None)
    c2 = Node(None,None)
    c1.isleaf= True
    c2.isleaf= True
    
    c1.p = 'c1' 
    c2.p = 'c2' 
    t.C = {'high':c1, 'low':c2}

    X = np.array([['apple','apple','apple','banana'],
                  ['high','low','low','high'],
                  ['a','b','c','a']])
    Y = Tree.predict(t,X)

    assert type(Y) == np.ndarray
    assert Y.shape == (4,) 
    assert Y[0] == 'c1'
    assert Y[1] == 'c2'
    assert Y[2] == 'c2'
    assert Y[3] == 'c1'


#-------------------------------------------------------------------------
def test_load_dataset():
    '''(4.5 points) load_dataset'''
    X, Y = Tree.load_dataset()
    assert type(X) == np.ndarray
    assert type(Y) == np.ndarray
    assert X.shape ==(7,42) 
    assert Y.shape ==(42,) 
    assert Y[0] == 'Bad'
    assert Y[1] == 'Bad'
    assert Y[4] == 'Good'
    assert Y[-1] == 'Bad'
    assert Y[-2] == 'Good'
    assert X[0,0] =='8'
    assert X[0,-1] =='6'
    assert X[1,0] =='201 to 400'
    assert X[1,-2] =='79 to 100'
    assert X[-1,0] =='America'
    assert X[-1,-2] =='Asia'


#-------------------------------------------------------------------------
def test_dataset1():
    '''(4.5 points) test_dataset1'''
    X, Y = Tree.load_dataset()
    t = Tree.train(X,Y) 
    Y_predict = Tree.predict(t,X) 
    accuracy = sum(Y==Y_predict)/42. # training accuracy of 42 training samples
    print('training accuracy:', accuracy)
    assert accuracy >= 39./42 # test training accuracy of 42 training samples

    # train over half of the dataset
    t = Tree.train(X[:,::2],Y[::2]) 
    # test on the other half
    Y_predict = Tree.predict(t,X[:,1::2]) 
    accuracy = sum(Y[1::2]==Y_predict)/21. 
    print( 'test accuracy:', accuracy)
    assert accuracy >= .7
