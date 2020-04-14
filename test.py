import numpy as np
import pickle
import math
import os

class pagerank:

    def __init__(self,top,beta,basketsize):
        self.originDataPath = 'data/WikiData.txt'
        self.sortedDataPath = 'data/mappedtest_WikiData.txt'
        self.resultDataPath = 'data/result.txt'
        self.top=top
        self.beta=beta
        self.nodes = self.read_file()
        self.basketsize=basketsize
        self.basketnum=int(math.ceil(float(len(self.nodes)) / basketsize))
        print(
            self.basketnum
        )
        self.nodenum=self.sort_node()
        print('nodenum%d'%self.nodenum)
        self.sort_data()
        self.to_blockmatrix()
        self.generate_top()

    def read_file(self):
        originData = np.loadtxt(self.originDataPath,dtype='int')
        nodes = np.hstack((originData[:, 0],originData[:, 1]))
        nodes = np.unique(nodes)
        nodes.sort()
        return nodes

    def sort_node(self):
        nodenum = len(self.nodes)
        index = [i for i in range(nodenum)]
        map = dict(zip(index, self.nodes))
        maprev=dict(zip(self.nodes, index))
        pickle.dump(map, open('data/mid/mapor.txt', 'wb+'))
        pickle.dump(maprev, open('data/mid/maprev.txt', 'wb+'))
        return nodenum

    def sort_data(self):
        mapor = pickle.load(open('data/mid/maprev.txt', 'rb'))
        originData = np.loadtxt(self.originDataPath, dtype='int')
        dist=[]
        for i in range(originData.shape[0]):
            tmp = originData[i]
            l1=mapor.get(tmp[0])
            l2 = mapor.get(tmp[1])
            dist.append([l1,l2])
        np.savetxt(self.sortedDataPath,dist,fmt="%d %d")

    def to_blockmatrix(self):
        sortedData=np.loadtxt(self.sortedDataPath,dtype='int')
        dist = []
        for i in range(sortedData.shape[0]):
            if (i+1)%1000 == 0:
                print("matrix to block finished :")
                print(i*1.0/sortedData.shape[0])
            tmp = sortedData[i] 
            if i == sortedData.shape[0]-1:
                dist.append(tmp[1])
                degree = len(dist)
                blocks = [[degree] for _ in range(self.basketnum)]
                for item in dist:
                    blocks[int(item / self.basketsize)].append(item)
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        np.savetxt(('data/mid/blocks_%d_%d.txt' % (tmp[0], bas)),blocks[bas])
                dist=[]
            elif sortedData[i+1,0] != sortedData[i,0]:
                dist.append(tmp[1])
                degree = len(dist)
                blocks = [[degree] for _ in range(self.basketnum)]
                for item in dist:
                    blocksid=int(item / self.basketsize)
                    blocks[blocksid].append(item)
                for bas in range(self.basketnum):
                    if len(blocks[bas]) > 1:
                        np.savetxt(('data/mid/blocks_%d_%d.txt' % (tmp[0], bas)), blocks[bas])
                dist=[]
            else:
                dist.append(tmp[1])

    def generate_top(self):
        for item in range(self.basketnum):
            r = [ 1.0 / (self.nodenum) for _ in range(self.basketsize)]
            np.save('data/oldr/oldr_%d.npy' % item, r)
        e = 1
        i=0
        print('origin sum:',1.0 / (self.nodenum)*self.basketsize*self.basketnum)
        # print result
        while e > 1e-6:
            print(str(i)+" time train "+str(e))
            e = 0
            # item 
            # print('self.nodenum:%d'%self.nodenum)
            solvedeadend = 0;
            for item in range(self.basketnum): 
                r_new = np.array([(1.0 - beta) / self.nodenum for _ in range(self.basketsize)])
                for src in range(self.nodenum):
                    r_old = np.load('data/oldr/oldr_%d.npy' % (int(src/self.basketsize)))
                    if not os.path.exists('data/mid/blocks_%d_%d.txt' % (src, item)):
                        continue
                    else:
                        line = np.loadtxt('data/mid/blocks_%d_%d.txt' % (src, item))
                        di = line[0]
                        destList = [nodes for nodes in line[1:]]
                        # print('src:',src,'destList',destList,'di',di)
                        for k in destList:
                            r_new[int(k % self.basketsize )] += beta * r_old[int(src % self.basketsize)] / di
                            # print('r_old[int(src % self.basketsize)] / di',r_old[int(src % self.basketsize)] / di)
                np.save('data/newr/newr_%d.npy' % item, r_new)   
            for src in range(self.nodenum):
                judge=0
                for item in range(self.basketnum): 
                    if os.path.exists('data/mid/blocks_%d_%d.txt' % (src, item)):
                        judge=1
                if judge == 0:
                    r_old = np.load('data/oldr/oldr_%d.npy' % (int(src/self.basketsize)))
                    solvedeadend += beta * r_old[int(src % self.basketsize)]/self.nodenum
            print('solvedeadend',solvedeadend)
            for i in range(self.basketnum):
                rn = np.load('data/newr/newr_%d.npy' % i)
                rn = [i+solvedeadend for i in rn]
                np.save('data/newr/newr_%d.npy' % i, rn)     
            x = []
            for i in range(self.basketnum):
                r = np.load('data/newr/newr_%d.npy' % i)
                for item in r:
                    x.append(item)
                    x = x[:self.nodenum]
            print('sum(x):',sum(x))
            for i in range(self.basketnum):
                rn = np.load('data/newr/newr_%d.npy' % i)
                ro = np.load('data/oldr/oldr_%d.npy' % i)
                e += np.linalg.norm((np.array(rn) - np.array(ro)), ord=1)
                np.save('data/oldr/oldr_%d.npy' % i, rn)
        print('x',x)
        print('sum',sum(x))
        temp=sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:self.top]
        score=sorted(x,reverse=True)[:self.top];
        mapor = pickle.load(open('data/mid/mapor.txt', 'rb'))
        print('self.top%d'%self.top)
        re=[]
        for i in range(self.top):
            l1=mapor.get(temp[i]) # nodeid
            re.append([int(l1),score[i]])
            print('nodeid: %d   score[%d]: %f'%(l1,i,score[i]))
        np.savetxt(self.resultDataPath,re, fmt="%d %f")


if __name__ == '__main__':
    top = 100
    beta = 0.85
    basketsize= 700
    p = pagerank(top, beta,basketsize)
