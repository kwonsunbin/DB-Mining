import pymysql
import csv
import time
import pandas as pd
from sklearn import tree
import graphviz
from sqlalchemy import create_engine
import sqlite3
import numpy as np
from scipy.sparse import lil_matrix  # other types would convert to LIL anyway
from orangecontrib.associate.fpgrowth import *

def association(localhost, root, password):

    # Create 'db2017_10' and Conncect to 'db2017_10'
    conn = pymysql.connect(host=localhost,
                           user=root,
                           password=password)
    curs = conn.cursor()
    sql = 'CREATE DATABASE IF NOT EXISTS db2017_10'
    curs.execute(sql)
    conn.commit()
    print('Create DATABASE db2017_10 successfully!!')
    conn.close()
    
    conn = pymysql.connect(host=localhost,
                           user=root,
                           password=password,
                           db='db2017_10',
                           charset='utf8',
                           cursorclass=pymysql.cursors.DictCursor)
    curs = conn.cursor()
    print("Connect to db2017_10 Successfully!!")
    """
    # data 전처리 과정
    questionposts = pd.read_csv('./dataset/questionPosts.csv', sep = ',', encoding = 'windows-1252')
    tag_name = pd.read_csv('./dataset/tagname.csv', sep = ',')

    tag_name1 = tag_name['Tag'][:100]
    tag_name2 = "Tags에'<"+tag_name1+">'이 포함되어 있음 (0 혹은 1)"
    tag_name1 = "<"+tag_name1+">"

    question_post_id = questionposts['Id']
    question_post_tags = questionposts['Tags']

    ##type change
    question_post_id1 = question_post_id.astype('str')

    question_post_id1 = "id "+question_post_id1

    df = pd.DataFrame(index=question_post_id1, columns=tag_name2)
    df.fillna(0)

    for i in range(1, len(df) + 1):
        print("uploading...", i / len(df) * 100, "%")
        for j in range(0, len(tag_name1)):
            if tag_name1[j] in question_post_tags[i - 1]:
                df[tag_name2[j]][i - 1] = 1
            else:
                df[tag_name2[j]][i - 1] = 0

    print("Complete!")
    
    df.to_csv("./dataset/TagMatrix.csv", index=True)
    """
    conn.close()
    
    df = pd.read_csv("./dataset/TagMatrix.csv")
    engine = create_engine("mysql+pymysql://root:017330@localhost:3306/db2017_10?charset=utf8", encoding ='utf-8')
    connected_engine = engine.connect()
    
    #TagMatrix.csv 파일을 SQL 상에 table로 생성
    df.to_sql(name="TagMatrix", con=connected_engine, if_exists='replace', index=False)

    conn = pymysql.connect(host=localhost,
                           user=root,
                           password=password,
                           db='db2017_10',
                           charset='utf8',
                           cursorclass=pymysql.cursors.DictCursor)
    
    #CREATE TagMatrix view
    sql = '''
        CREATE OR REPLACE VIEW TagMatrix AS
        SELECT * FROM TagMatrix'''
    curs.execute(sql)
    conn.commit

    #Association analysis
    #loading view using pandas
    data = pd.read_sql('SELECT * FROM TagMatrix', con=conn)
    tag_name = pd.read_csv('./dataset/tagname.csv', sep=',')

    tag_name1 = tag_name['Tag'][:100]
    tag_name2 = "Tags에'<" + tag_name1 + ">'이 포함되어 있음 (0 혹은 1)"
    tag_name1 = "<" + tag_name1 + ">"

    for i in range(0, len(tag_name2)):
        data[tag_name2[i]] = data[tag_name2[i]].astype(bool)

    data = data.drop('Id', 1)

    data = data.as_matrix()

    item_sets = dict(frequent_itemsets(data, 0.01))
    # frequent item sets that satisfy 0.01 min support
    # that is, how frequently the itemset appears in the dataset

    # among itemsets that satisfy the min_support, find all the itemsets with at least 0.05 confidence
    rules = association_rules(item_sets, 0.05)
    # what are the rules?
    # {0} means the name of the col in the data
    rules = list(rules)
    a = len(rules)

    for i in range(0,a):
        print(list(rules_stats(rules, item_sets, 42921))[i])

    conn.close()


def decisiontree1(localhost, root, password):
    # Create 'db2017_10' and Conncect to 'db2017_10'
    conn = pymysql.connect(host=localhost,
                           user=root,
                           password=password)
    curs = conn.cursor()
    sql = 'CREATE DATABASE IF NOT EXISTS db2017_10'
    curs.execute(sql)
    conn.commit()
    print('Create DATABASE db2017_10 successfully!!')
    conn.close()
    conn = pymysql.connect(host=localhost,
                           user=root,
                           password=password,
                           db='db2017_10',
                           charset='utf8',
                           cursorclass=pymysql.cursors.DictCursor)
    curs = conn.cursor()
    print("Connect to db2017_10 Successfully!!")

    # sql sentence(CREATE TABLE)
    tnow1 = time.time()
    sql = [' '] * 4

    # CREATE TABLE userInfo
    sql[0] = '''
            CREATE TABLE IF NOT EXISTS userInfo (
    	    UId INT(11) NOT NULL,
    	    Reputation INT(11) NOT NULL,
    	    DisplayName VARCHAR(255) NOT NULL,
    	    Age INT(11),
    	    CreationDate DATETIME NOT NULL,
    	    LastAccessDate DATETIME NOT NULL,
    	    WebsiteUrl VARCHAR(255),
    	    Location VARCHAR(255),
    	    AboutMe LONGTEXT,

    	    PRIMARY KEY(UId)
        ) ENGINE = InnoDB DEFAULT CHARSET = utf8
        '''
    # CREATE TABLE posts
    sql[1] = '''
            CREATE TABLE IF NOT EXISTS posts (
    	    PId INT(11) NOT NULL,
    	    CreationDate DATETIME NOT NULL,
    	    Body LONGTEXT NOT NULL,
    	    OwnerUserId INT(11) NOT NULL,
    	    LasActivityDate DATETIME NOT NULL,

    	    PRIMARY KEY(PId),
    	    FOREIGN KEY(OwnerUserId) REFERENCES userInfo(UId) ON DELETE CASCADE
        ) ENGINE = InnoDB DEFAULT CHARSET = utf8
        '''
    # CREATE TABLE badges
    sql[2] = '''
        CREATE TABLE IF NOT EXISTS badges (
	    BId INT(11) NOT NULL,
	    UserInfoId INT(11) NOT NULL,
	    Name VARCHAR(255) NOT NULL,
	    Date DATETIME NOT NULL,

	    PRIMARY KEY(BId),
	    FOREIGN KEY(UserInfoId) REFERENCES userInfo(UId) ON DELETE CASCADE
    ) ENGINE = InnoDB DEFAULT CHARSET = utf8
'''
    # CREATE TABLE comments
    sql[3] = '''
        CREATE TABLE IF NOT EXISTS comments (
	    CId INT(11) NOT NULL,
	    PostId INT(11) NOT NULL,
	    Score INT(11) NOT NULL,
	    CreationDate DATETIME NOT NULL,
	    UserInfoId INT(11) NOT NULL,

	    PRIMARY KEY(CId),
	    FOREIGN KEY(PostId) REFERENCES posts(PId) ON DELETE CASCADE,
	    FOREIGN KEY(UserInfoId) REFERENCES userInfo(UId) ON DELETE CASCADE
    ) ENGINE = InnoDB DEFAULT CHARSET = utf8
    '''

    # sql execution
    for i in range(0, 4):
        curs.execute(sql[i])
        conn.commit
    tnow2 = time.time()
    print("CREATE TABLE FINISH : It took ", tnow2-tnow1)

    # INSERT INTO userInfo
    tnow3 = time.time()
    f = open('./dataset/userInfo.csv', 'r', encoding='utf-8', errors='replace')
    rdr = csv.reader(f)
    next(rdr, None)
    userInfo = []

    for line in rdr:
        for i in (0, 1, 3):
            if line[i] != "":
                line[i] = int(line[i])
            else:
                line[i] = None
        for j in (2, 4, 5, 6, 7, 8):
            if line[j] == "":
                line[j] = None
        userInfo.append(line)

    f.close()

    sql = '''
        INSERT IGNORE INTO userInfo(UId, Reputation, DisplayName, Age, CreationDate, LastAccessDate, WebsiteUrl, Location, AboutMe)
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)'''
    curs.executemany(sql, userInfo)
    conn.commit()
    # INSERT INTO posts
    f = open('./dataset/posts.csv', 'r', encoding='utf-8', errors='replace')
    rdr = csv.reader(f)
    next(rdr, None)
    posts = []

    for line in rdr:
        for i in (0, 3):
            if line[i] != "":
                line[i] = int(line[i])
            else:
                line[i] = None
        for j in (1, 2, 4):
            if line[j] == "":
                line[j] = None
        posts.append(line)

    f.close()

    sql = '''
        INSERT IGNORE INTO posts(PId, CreationDate, Body, OwnerUserId, LasActivityDate)
        VALUES(%s, %s, %s, %s, %s)'''
    curs.executemany(sql, posts)
    conn.commit()
    # INSERT INTO badges
    f = open('./dataset/badges.csv', 'r', encoding='utf-8', errors='replace')
    rdr = csv.reader(f)
    next(rdr, None)
    badges = []

    for line in rdr:
        for i in (0, 1):
            if line[i] != "":
                line[i] = int(line[i])
            else:
                line[i] = None
        for j in (2, 3):
            if line[j] == "":
                line[j] = None
        badges.append(line)

    f.close()

    sql = '''
        INSERT IGNORE INTO badges(BId, UserInfoId, Name, Date)
        VALUES(%s, %s, %s, %s)'''
    curs.executemany(sql, badges)
    conn.commit()
    # INSERT INTO comments
    f = open('./dataset/comments.csv', 'r', encoding='utf-8', errors='replace')
    rdr = csv.reader(f)
    next(rdr, None)
    comments = []

    for line in rdr:
        for i in (0, 1, 2, 4):
            if line[i] != "":
                line[i] = int(line[i])
            else:
                line[i] = None
        for j in range(3, 4):
            if line[j] == "":
                line[j] = None
        comments.append(line)

    f.close()

    sql = '''
        INSERT IGNORE INTO comments(CId, PostId, Score, CreationDate, UserInfoId)
        VALUES(%s, %s, %s, %s, %s)'''
    curs.executemany(sql, comments)
    conn.commit()

    tnow4 = time.time()
    print("INSERT INTO data COMPLETE : It took ", tnow4-tnow3)

    #CREATE ReputStatMatrix view
    tnow5 = time.time()
    sql = '''
        CREATE OR REPLACE VIEW ReputStatMatrix AS
        SELECT UserId, Reputation, IFNULL(NumOfPosts,0) AS NumOfPosts, IFNULL(NumOfComments,0) AS NumOfComments, IFNULL(NumOfBadges,0) AS NumOfBadges
        FROM(
        SELECT userInfo.UId AS UserId, userInfo.Reputation
        FROM userInfo
        WHERE userInfo.Reputation>110) AS Temp1
        LEFT JOIN(
        SELECT OwnerUserId, COUNT(*) AS NumOfPosts
        FROM posts
        GROUP BY OwnerUserId) AS Temp2
        ON UserId = OwnerUserId
        LEFT JOIN(
        SELECT UserInfoId AS UId1, COUNT(*) AS NumOfComments
        FROM comments
        GROUP BY UId1) AS Temp3
        ON UserId = UId1
        LEFT JOIN(
        SELECT UserInfoId AS UId2, COUNT(*) AS NumOfBadges
        FROM badges
        GROUP BY UId2) AS Temp4
        ON UserId = UId2'''
    curs.execute(sql)
    conn.commit
    tnow6 = time.time()
    print("CREATE VIEW ReputStatMatrix COMPLETE : It took ", tnow6 - tnow5)

    #loading view using pandas
    tnow7 = time.time()
    df = pd.read_sql('SELECT * FROM ReputStatMatrix', con=conn)

    #preprocessing: 'Reputation' > 180 is classified 1 class, otherwise 0 class
    df['Reputation'] = (df['Reputation'] > 180).astype(int)
    features = list(df[['NumOfPosts', 'NumOfComments', 'NumOfBadges']])
    print("User whose Reputation is above 180 is classified 1 class, otherwise 0 class.")
    feature_name = ["NumOfPosts", "NumOfComments", "NumOfBadges"]
    target_name = ["Reputation below 180", "Reputation above 180"]

    #fitting the decision tree
    y = df['Reputation']
    x = df[features]
    dt = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=10)
    dt = dt.fit(x,y)

    # visualizing the tree
    dot_data = tree.export_graphviz(dt, out_file=None, feature_names=feature_name, class_names=target_name, filled=True)
    graph = graphviz.Source(dot_data, format = 'png')
    graph.render('decisiontree1_gini')

    #predict class and probability
    predict_data = [[5, 5, 5], [2, 6, 18], [6, 3, 10]]
    for i in range(0, 3):
        print("Class of UserId", 1000000+i, "is", dt.predict([predict_data[i]]))
        print("The classifying probability of UserId", 1000001+i, "is", dt.predict_proba([predict_data[i]]), "(0 class, 1 class in order)")
    tnow8 = time.time()
    print("'gini' Decision Tree COMPLETE : It took ", tnow8 - tnow7, '\n')

    #fitting the decision tree in 'entropy' criterion
    tnow9 = time.time()
    dt = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=10)
    dt = dt.fit(x,y)

    #visualizing the tree
    dot_data = tree.export_graphviz(dt, out_file=None, feature_names=feature_name, class_names=target_name, filled=True)
    graph = graphviz.Source(dot_data, format = 'png')
    graph.render('decisiontree1_entropy')

    #predict class and probability
    for i in range(0, 3):
        print("Class of UserId", 1000000+i, "is", dt.predict([predict_data[i]]))
        print("The classifying probability of UserId", 1000001+i, "is", dt.predict_proba([predict_data[i]]), "(0 class, 1 class in order)")
    tnow10 = time.time()
    print("'entropy' Decision Tree COMPLETE : It took ", tnow10 - tnow9, '\n')

    conn.close()


def decisiontree2(localhost, root, password):
    # Create 'db2017_10' and Conncect to 'db2017_10'
    conn = pymysql.connect(host=localhost,
                           user=root,
                           password=password)
    curs = conn.cursor()
    sql = 'CREATE DATABASE IF NOT EXISTS db2017_10'
    curs.execute(sql)
    conn.commit()
    print('Create DATABASE db2017_10 successfully!!')
    conn.close()
    conn = pymysql.connect(host=localhost,
                           user=root,
                           password=password,
                           db='db2017_10',
                           charset='utf8',
                           cursorclass=pymysql.cursors.DictCursor)
    curs = conn.cursor()
    print("Connect to db2017_10 Successfully!!")

    # sql sentence(CREATE TABLE)
    tnow1 = time.time()
    sql = [' '] * 2

    # CREATE TABLE postHistory
    sql[0] = '''
            CREATE TABLE IF NOT EXISTS postHistory (
    	    HId INT(11) NOT NULL,
    	    PostHistoryTypeId INT(11) NOT NULL,
    	    PostId INT(11) NOT NULL,
    	    CreationDate DATETIME NOT NULL,
    	    UserInfoId INT(11) NOT NULL,
    	    Text LONGTEXT,
    	    Comment LONGTEXT,

    	    PRIMARY KEY(HId),
    	    FOREIGN KEY(PostId) REFERENCES posts(PId) ON DELETE CASCADE,
    	    FOREIGN KEY(UserInfoId) REFERENCES userInfo(UId) ON DELETE CASCADE
        ) ENGINE = InnoDB DEFAULT CHARSET = utf8
        '''
    # CREATE TABLE votes
    sql[1] = '''
            CREATE TABLE IF NOT EXISTS votes (
    	    VId INT(11) NOT NULL,
    	    PostId INT(11) NOT NULL,
    	    VoteTypeId INT(11) NOT NULL,
    	    CreationDate DATE NOT NULL,
    	    UserInfoId INT(11),
    	    BountyAmount INT(11),

    	    PRIMARY KEY(VId),
    	    FOREIGN KEY(PostId) REFERENCES posts(PId) ON DELETE CASCADE
        ) ENGINE = InnoDB DEFAULT CHARSET = utf8
        '''

    # sql execution
    for i in range(0, 2):
        curs.execute(sql[i])
        conn.commit
    tnow2 = time.time()
    print("CREATE TABLE FINISH : It took ", tnow2-tnow1)

    # INSERT INTO postHistory
    tnow3 = time.time()
    f = open('./dataset/postHistory.csv', 'r', encoding='utf-8', errors='replace')
    rdr = csv.reader(f)
    next(rdr, None)
    postHistory = []

    for line in rdr:
        for i in (0, 1, 2, 4):
            if line[i] != "":
                line[i] = int(line[i])
            else:
                line[i] = None
        for j in (3, 5, 6):
            if line[j] == "":
                line[j] = None
        postHistory.append(line)

    f.close()

    sql = '''
        INSERT IGNORE INTO postHistory(HId, PostHistoryTypeId, PostId, CreationDate, UserInfoId, Text, Comment)
        VALUES(%s, %s, %s, %s, %s, %s, %s)'''
    curs.executemany(sql, postHistory)
    conn.commit()

    # INSERT INTO votes
    f = open('./dataset/votes.csv', 'r', encoding='utf-8', errors='replace')
    rdr = csv.reader(f)
    next(rdr, None)
    votes = []

    for line in rdr:
        for i in (0, 1, 2, 4, 5):
            if line[i] != "":
                line[i] = int(line[i])
            else:
                line[i] = None
        votes.append(line)

    f.close()

    sql = '''
        INSERT IGNORE INTO votes(VId, PostId, VoteTypeId, CreationDate, UserInfoId, BountyAmount)
        VALUES(%s, %s, %s, %s, %s, %s)'''
    curs.executemany(sql, votes)
    conn.commit()

    tnow4 = time.time()
    print("INSERT INTO data COMPLETE : It took ", tnow4-tnow3)

    #CREATE ReputStatMatrix2 view
    tnow5 = time.time()
    sql = '''
        CREATE OR REPLACE VIEW ReputStatMatrix2 AS
        SELECT UserId, Reputation, IFNULL(NumOfPosts,0) AS NumOfPosts, IFNULL(NumOfComments,0) AS NumOfComments, IFNULL(NumOfBadges,0) AS NumOfBadges, IFNULL(NumOfPostHistorys,0) AS NumOfPostHistorys, IFNULL(NumOfVotes,0) AS NumOfVotes
        FROM(
        SELECT userInfo.UId AS UserId, userInfo.Reputation
        FROM userInfo
        WHERE userInfo.Reputation>110) AS Temp1
        LEFT JOIN(
        SELECT OwnerUserId, COUNT(*) AS NumOfPosts
        FROM posts
        GROUP BY OwnerUserId) AS Temp2
        ON UserId = OwnerUserId
        LEFT JOIN(
        SELECT UserInfoId AS UId1, COUNT(*) AS NumOfComments
        FROM comments
        GROUP BY UId1) AS Temp3
        ON UserId = UId1
        LEFT JOIN(
        SELECT UserInfoId AS UId2, COUNT(*) AS NumOfBadges
        FROM badges
        GROUP BY UId2) AS Temp4
        ON UserId = UId2
        LEFT JOIN(
        SELECT UserInfoId AS UId3, COUNT(*) AS NumOfPostHistorys
        FROM postHistory
        GROUP BY UId3) AS Temp5
        ON UserId = UId3
        LEFT JOIN(
        SELECT UserInfoId AS UId4, COUNT(*) AS NumOfVotes
        FROM votes
        GROUP BY UId4) AS Temp6
        ON UserId = UId4'''
    curs.execute(sql)
    conn.commit
    tnow6 = time.time()
    print("CREATE VIEW ReputStatMatrix2 COMPLETE : It took ", tnow6 - tnow5)

    #loading view using pandas
    tnow7 = time.time()
    df = pd.read_sql('SELECT * FROM reputstatmatrix2', con=conn)

    #preprocessing: 'Reputation' > 180 is classified 1 class, otherwise 0 class
    df['Reputation'] = (df['Reputation'] > 180).astype(int)
    features = list(df[['NumOfPosts', 'NumOfComments', 'NumOfBadges', 'NumOfPostHistorys', 'NumOfVotes']])
    print("User whose Reputation is above 180 is classified 1 class, otherwise 0 class.")
    feature_name = ["NumOfPosts", "NumOfComments", "NumOfBadges", "NumOfPostHistorys", "NumOfVotes"]
    target_name = ["Reputation below 180", "Reputation above 180"]

    #fitting the decision tree
    y = df['Reputation']
    x = df[features]
    dt = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=10)
    dt = dt.fit(x,y)

    # visualizing the tree
    dot_data = tree.export_graphviz(dt, out_file=None, feature_names=feature_name, class_names=target_name, filled=True)
    graph = graphviz.Source(dot_data, format = 'png')
    graph.render('decisiontree2_gini')
    tnow8 = time.time()
    print("'gini' Decision Tree COMPLETE : It took ", tnow8 - tnow7, '\n')

    #fitting the decision tree in 'entropy' criterion
    tnow9 = time.time()
    dt = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=10)
    dt = dt.fit(x,y)

    #visualizing the tree
    dot_data = tree.export_graphviz(dt, out_file=None, feature_names=feature_name, class_names=target_name, filled=True)
    graph = graphviz.Source(dot_data, format = 'png')
    graph.render('decisiontree2_entropy')
    tnow10 = time.time()
    print("'entropy' Decision Tree COMPLETE : It took ", tnow10 - tnow9, '\n')

    conn.close()


association('localhost', 'root', '017330')
decisiontree1('localhost','root','017330')
decisiontree2('localhost','root','017330')